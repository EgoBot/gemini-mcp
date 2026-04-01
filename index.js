import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import os from "os";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

// ── Usage Logging ───────────────────────────────────────────

const USAGE_LOG_PATH = path.join(os.homedir(), "Desktop", "gemini-outputs", "usage-log.jsonl");

// Pricing table (USD) — update if Google changes rates
const PRICING = {
  "imagen-4.0-generate-001": { "1K": 0.04, "2K": 0.06 },          // per image
  "gemini-3.1-flash-image-preview": { "512": 0.04, "1K": 0.04, "2K": 0.134, "4K": 0.24 }, // per image
  "veo-3.1-generate-preview": { perSecond: 0.15 },                 // per second (fast, with audio)
};

function logUsage({ tool, model, params, outputCount, estimatedCost }) {
  const entry = {
    timestamp: new Date().toISOString(),
    tool,
    model,
    params,
    outputCount,
    estimatedCost: Math.round(estimatedCost * 1000) / 1000,
  };
  try {
    fs.mkdirSync(path.dirname(USAGE_LOG_PATH), { recursive: true });
    fs.appendFileSync(USAGE_LOG_PATH, JSON.stringify(entry) + "\n");
  } catch (e) {
    // Don't fail the generation if logging fails
  }
}

// ── Helpers ─────────────────────────────────────────────────

function getOutputDir() {
  const primary = path.join(os.homedir(), "Desktop", "gemini-outputs");
  const fallback = path.join(os.homedir(), "Downloads", "gemini-outputs");
  for (const dir of [primary, fallback]) {
    try {
      fs.mkdirSync(dir, { recursive: true });
      return dir;
    } catch {
      continue;
    }
  }
  throw new Error("Cannot create output directory in ~/Desktop or ~/Downloads");
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

function readImageAsBase64(filePath) {
  const resolved = filePath.replace(/^~/, os.homedir());
  if (!fs.existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }
  const bytes = fs.readFileSync(resolved);
  const ext = path.extname(resolved).toLowerCase();
  const mimeMap = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
  };
  return {
    data: bytes.toString("base64"),
    mimeType: mimeMap[ext] || "image/png",
  };
}

async function geminiRequest(endpoint, body) {
  const url = `${BASE_URL}${endpoint}?key=${GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${err}`);
  }
  return res.json();
}

async function geminiGet(endpoint) {
  const url = `${BASE_URL}/${endpoint}?key=${GEMINI_API_KEY}`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${err}`);
  }
  return res.json();
}

// ── MCP Server ──────────────────────────────────────────────

const server = new McpServer({
  name: "gemini-mcp",
  version: "2.0.0",
});

// ── Tool: Generate Image (Imagen 4) ────────────────────────

server.tool(
  "gemini_generate_image",
  "Generate images using Google Imagen 4. Supports text-to-image and image references (style/subject). Saves PNG to ~/Desktop/gemini-outputs/.",
  {
    prompt: z.string().describe("Text description of the image to generate"),
    aspectRatio: z
      .enum(["1:1", "3:4", "4:3", "9:16", "16:9"])
      .default("1:1")
      .describe("Aspect ratio"),
    sampleCount: z
      .number()
      .int()
      .min(1)
      .max(4)
      .default(1)
      .describe("Number of images to generate (1-4)"),
    imageSize: z
      .enum(["1K", "2K"])
      .default("1K")
      .describe("Output resolution: 1K or 2K"),
    referenceImagePath: z
      .string()
      .optional()
      .describe("Optional: path to a reference image for style or subject guidance"),
    personGeneration: z
      .enum(["dont_allow", "allow_adult", "allow_all"])
      .default("allow_adult")
      .describe("Person/face generation policy"),
    seed: z
      .number()
      .int()
      .optional()
      .describe("Optional: seed for reproducible generation (disables watermark)"),
  },
  async ({ prompt, aspectRatio, sampleCount, imageSize, referenceImagePath, personGeneration, seed }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      const instance = { prompt };

      // Add reference image if provided
      if (referenceImagePath) {
        const img = readImageAsBase64(referenceImagePath);
        instance.image = {
          bytesBase64Encoded: img.data,
          mimeType: img.mimeType,
        };
      }

      const parameters = {
        sampleCount,
        aspectRatio,
        imageSize,
        personGeneration,
      };

      // Seed requires watermark off
      if (seed !== undefined) {
        parameters.seed = seed;
        parameters.addWatermark = false;
      }

      const data = await geminiRequest(
        "/models/imagen-4.0-generate-001:predict",
        { instances: [instance], parameters }
      );

      // Handle both Imagen 4 and legacy response formats
      const images = data.generatedImages || data.predictions;
      if (!images || images.length === 0) {
        return {
          content: [{ type: "text", text: "No images returned. The prompt may have been blocked by safety filters." }],
        };
      }

      const outputDir = getOutputDir();
      const savedFiles = [];

      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const b64 = img.image?.imageBytes || img.bytesBase64Encoded;
        if (!b64) continue;

        const buffer = Buffer.from(b64, "base64");
        const filename = `imagen4-${timestamp()}-${i + 1}.png`;
        const filepath = path.join(outputDir, filename);
        fs.writeFileSync(filepath, buffer);
        savedFiles.push(filepath);
      }

      if (savedFiles.length === 0) {
        return { content: [{ type: "text", text: "Images were returned but could not be decoded." }] };
      }

      // Log usage
      const pricePerImage = PRICING["imagen-4.0-generate-001"][imageSize] || 0.04;
      logUsage({
        tool: "gemini_generate_image",
        model: "imagen-4.0-generate-001",
        params: { sampleCount, aspectRatio, imageSize, personGeneration, hasReference: !!referenceImagePath },
        outputCount: savedFiles.length,
        estimatedCost: savedFiles.length * pricePerImage,
      });

      return {
        content: [{ type: "text", text: `Generated ${savedFiles.length} image(s):\n${savedFiles.join("\n")}` }],
      };
    } catch (err) {
      return { content: [{ type: "text", text: `Error: ${err.message}` }] };
    }
  }
);

// ── Tool: Native Image Gen/Edit (Nano Banana) ──────────────

server.tool(
  "gemini_native_image",
  "Generate or edit images using Gemini 3.1 Flash Image (Nano Banana). Supports multiple input images, image editing, multi-turn style, and up to 4K. Saves PNG to ~/Desktop/gemini-outputs/.",
  {
    prompt: z.string().describe("Text description or editing instruction"),
    aspectRatio: z
      .enum(["1:1", "1:4", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])
      .default("1:1")
      .describe("Aspect ratio (14 options)"),
    imageSize: z
      .enum(["512", "1K", "2K", "4K"])
      .default("1K")
      .describe("Output resolution"),
    inputImagePaths: z
      .array(z.string())
      .optional()
      .describe("Optional: array of absolute paths to input images for editing, reference, or multi-image context"),
    personGeneration: z
      .enum(["dont_allow", "allow_adult", "allow_all"])
      .default("allow_adult")
      .describe("Person/face generation policy"),
  },
  async ({ prompt, aspectRatio, imageSize, inputImagePaths, personGeneration }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      const parts = [];

      // Add all input images
      if (inputImagePaths && inputImagePaths.length > 0) {
        for (const imgPath of inputImagePaths) {
          const img = readImageAsBase64(imgPath);
          parts.push({
            inline_data: {
              mime_type: img.mimeType,
              data: img.data,
            },
          });
        }
      }

      parts.push({ text: prompt });

      const data = await geminiRequest(
        "/models/gemini-3.1-flash-image-preview:generateContent",
        {
          contents: [{ role: "user", parts }],
          generationConfig: {
            responseModalities: ["TEXT", "IMAGE"],
            imageConfig: {
              aspectRatio,
              imageSize,
            },
          },
          safetySettings: [
            {
              category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold: personGeneration === "allow_all" ? "BLOCK_NONE" : "BLOCK_MEDIUM_AND_ABOVE",
            },
          ],
        }
      );

      const candidate = data.candidates?.[0];
      if (!candidate?.content?.parts) {
        // Debug: show what the API actually returned
        const debugInfo = JSON.stringify({
          hasData: !!data,
          hasCandidates: !!data?.candidates,
          candidateCount: data?.candidates?.length,
          finishReason: candidate?.finishReason,
          blockReason: data?.promptFeedback?.blockReason,
          safetyRatings: candidate?.safetyRatings || data?.promptFeedback?.safetyRatings,
        }, null, 2);
        return { content: [{ type: "text", text: `No content returned. Debug: ${debugInfo}` }] };
      }

      const outputDir = getOutputDir();
      const savedFiles = [];
      const textParts = [];

      for (const part of candidate.content.parts) {
        if (part.text) {
          textParts.push(part.text);
        }
        if (part.inline_data) {
          const buffer = Buffer.from(part.inline_data.data, "base64");
          const ext = part.inline_data.mime_type === "image/jpeg" ? "jpg" : "png";
          const filename = `nanob-${timestamp()}-${savedFiles.length + 1}.${ext}`;
          const filepath = path.join(outputDir, filename);
          fs.writeFileSync(filepath, buffer);
          savedFiles.push(filepath);
        }
      }

      if (savedFiles.length === 0) {
        const partTypes = candidate.content.parts.map(p => Object.keys(p).join(","));
        const msg = textParts.length > 0
          ? `No images generated. Model response: ${textParts.join(" ")} | Part types: ${partTypes.join("; ")}`
          : `No images returned. Part types found: ${partTypes.join("; ")} | finishReason: ${candidate.finishReason}`;
        return { content: [{ type: "text", text: msg }] };
      }

      // Log usage
      const nanoPricePerImage = PRICING["gemini-3.1-flash-image-preview"][imageSize] || 0.04;
      logUsage({
        tool: "gemini_native_image",
        model: "gemini-3.1-flash-image-preview",
        params: { aspectRatio, imageSize, personGeneration, inputImageCount: inputImagePaths?.length || 0 },
        outputCount: savedFiles.length,
        estimatedCost: savedFiles.length * nanoPricePerImage,
      });

      let result = `Generated ${savedFiles.length} image(s):\n${savedFiles.join("\n")}`;
      if (textParts.length > 0) {
        result += `\n\nModel notes: ${textParts.join(" ")}`;
      }

      return { content: [{ type: "text", text: result }] };
    } catch (err) {
      return { content: [{ type: "text", text: `Error: ${err.message}` }] };
    }
  }
);

// ── Tool: Generate Video (Veo 3.1) ─────────────────────────

server.tool(
  "gemini_generate_video",
  "Generate a video using Google Veo 3.1 with native audio. Supports text-to-video and image-to-video (start frame). Saves MP4 to ~/Desktop/gemini-outputs/.",
  {
    prompt: z.string().describe("Text description of the video to generate"),
    durationSeconds: z
      .enum(["4", "6", "8"])
      .default("6")
      .describe("Video duration: 4, 6, or 8 seconds"),
    aspectRatio: z
      .enum(["9:16", "16:9"])
      .default("16:9")
      .describe("Aspect ratio"),
    resolution: z
      .enum(["720p", "1080p", "4k"])
      .default("720p")
      .describe("Video resolution (1080p and 4k require 8s duration)"),
    inputImagePath: z
      .string()
      .optional()
      .describe("Optional: path to a start-frame image for image-to-video animation"),
    negativePrompt: z
      .string()
      .optional()
      .describe("Optional: describe unwanted elements as nouns (e.g. 'blur, text, watermark')"),
    personGeneration: z
      .enum(["allow_all"])
      .default("allow_all")
      .describe("Person/face generation policy (only allow_all is currently supported)"),
    seed: z
      .number()
      .int()
      .optional()
      .describe("Optional: seed for reproducible generation (0-4294967295)"),
  },
  async ({ prompt, durationSeconds, aspectRatio, resolution, inputImagePath, negativePrompt, personGeneration, seed }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    const duration = parseInt(durationSeconds, 10);

    if ((resolution === "1080p" || resolution === "4k") && duration !== 8) {
      return {
        content: [{ type: "text", text: `Error: ${resolution} resolution requires 8 second duration.` }],
      };
    }

    try {
      // Build instance
      const instance = { prompt };

      // Image-to-video: add start frame
      if (inputImagePath) {
        const img = readImageAsBase64(inputImagePath);
        instance.image = {
          bytesBase64Encoded: img.data,
          mimeType: img.mimeType,
        };
      }

      // Build parameters
      const parameters = {
        durationSeconds: duration,
        aspectRatio,
        resolution,
        personGeneration,
      };

      if (negativePrompt) {
        parameters.negativePrompt = negativePrompt;
      }

      if (seed !== undefined) {
        parameters.seed = seed;
      }

      // Step 1: Submit generation request
      const operation = await geminiRequest(
        "/models/veo-3.1-generate-preview:predictLongRunning",
        { instances: [instance], parameters }
      );

      if (!operation.name) {
        return { content: [{ type: "text", text: `Unexpected response: ${JSON.stringify(operation)}` }] };
      }

      // Step 2: Poll for completion (max 7 minutes)
      const maxWaitMs = 7 * 60 * 1000;
      const pollIntervalMs = 5000;
      const startTime = Date.now();
      let result;

      while (Date.now() - startTime < maxWaitMs) {
        await new Promise((r) => setTimeout(r, pollIntervalMs));
        result = await geminiGet(operation.name);
        if (result.done) break;
      }

      if (!result?.done) {
        return { content: [{ type: "text", text: "Video generation timed out after 7 minutes. Try again or use a simpler prompt." }] };
      }

      if (result.error) {
        return { content: [{ type: "text", text: `Generation failed: ${JSON.stringify(result.error)}` }] };
      }

      // Step 3: Download the video
      const samples = result.response?.generateVideoResponse?.generatedSamples;
      if (!samples || samples.length === 0) {
        return { content: [{ type: "text", text: "No video returned. The prompt may have been blocked by safety filters." }] };
      }

      const outputDir = getOutputDir();
      const savedFiles = [];

      for (let i = 0; i < samples.length; i++) {
        const videoUri = samples[i].video?.uri;
        if (!videoUri) continue;

        const videoRes = await fetch(videoUri, {
          headers: { "x-goog-api-key": GEMINI_API_KEY },
          redirect: "follow",
        });
        if (!videoRes.ok) {
          throw new Error(`Failed to download video: ${videoRes.status}`);
        }

        const videoBuffer = Buffer.from(await videoRes.arrayBuffer());
        const filename = `veo3-${timestamp()}-${i + 1}.mp4`;
        const filepath = path.join(outputDir, filename);
        fs.writeFileSync(filepath, videoBuffer);
        savedFiles.push(filepath);
      }

      // Log usage
      const veoPricePerSec = PRICING["veo-3.1-generate-preview"].perSecond || 0.15;
      logUsage({
        tool: "gemini_generate_video",
        model: "veo-3.1-generate-preview",
        params: { durationSeconds: duration, aspectRatio, resolution, personGeneration, hasStartFrame: !!inputImagePath },
        outputCount: savedFiles.length,
        estimatedCost: savedFiles.length * duration * veoPricePerSec,
      });

      return {
        content: [{ type: "text", text: `Generated ${savedFiles.length} video(s):\n${savedFiles.join("\n")}` }],
      };
    } catch (err) {
      if (err.message.includes("403") || err.message.includes("404")) {
        return {
          content: [{
            type: "text",
            text: `Veo 3.1 access error: ${err.message}\n\nThis likely means Veo 3.1 is not enabled for your API key. Check that your Gemini API plan supports video generation.`,
          }],
        };
      }
      return { content: [{ type: "text", text: `Error: ${err.message}` }] };
    }
  }
);

// ── Start ───────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});

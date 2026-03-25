import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import os from "os";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

// Output directory setup
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
  version: "1.2.0",
});

// ── Tool: Generate Image (Imagen 4) ────────────────────────

server.tool(
  "gemini_generate_image",
  "Generate an image using Google Imagen 4. Saves PNG to ~/Desktop/gemini-outputs/.",
  {
    prompt: z.string().describe("Text description of the image to generate"),
    aspectRatio: z
      .enum(["1:1", "3:4", "4:3", "9:16", "16:9"])
      .default("1:1")
      .describe("Aspect ratio of the generated image"),
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
  },
  async ({ prompt, aspectRatio, sampleCount, imageSize }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      const data = await geminiRequest(
        "/models/imagen-4.0-generate-001:predict",
        {
          instances: [{ prompt }],
          parameters: {
            sampleCount,
            aspectRatio,
            imageSize,
          },
        }
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
        // Imagen 4: generatedImages[].image.imageBytes
        // Legacy: predictions[].bytesBase64Encoded
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

      return {
        content: [
          {
            type: "text",
            text: `Generated ${savedFiles.length} image(s):\n${savedFiles.join("\n")}`,
          },
        ],
      };
    } catch (err) {
      return { content: [{ type: "text", text: `Error: ${err.message}` }] };
    }
  }
);

// ── Tool: Native Image Gen/Edit (Nano Banana) ──────────────

server.tool(
  "gemini_native_image",
  "Generate or edit images using Gemini 3.1 Flash Image (Nano Banana). Supports text-to-image, image editing via instructions, multi-turn refinement, and up to 4K output. Saves PNG to ~/Desktop/gemini-outputs/.",
  {
    prompt: z.string().describe("Text description or editing instruction for the image"),
    aspectRatio: z
      .enum(["1:1", "1:4", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])
      .default("1:1")
      .describe("Aspect ratio (more options than Imagen)"),
    imageSize: z
      .enum(["512", "1K", "2K", "4K"])
      .default("1K")
      .describe("Output resolution"),
    inputImagePath: z
      .string()
      .optional()
      .describe("Optional: absolute path to an image file to edit/modify"),
  },
  async ({ prompt, aspectRatio, imageSize, inputImagePath }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      // Build content parts
      const parts = [];

      // If an input image is provided, include it
      if (inputImagePath) {
        const imgPath = inputImagePath.replace(/^~/, os.homedir());
        if (!fs.existsSync(imgPath)) {
          return { content: [{ type: "text", text: `Error: File not found: ${imgPath}` }] };
        }
        const imgBytes = fs.readFileSync(imgPath);
        const ext = path.extname(imgPath).toLowerCase();
        const mimeMap = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif" };
        const mimeType = mimeMap[ext] || "image/png";
        parts.push({
          inline_data: {
            mime_type: mimeType,
            data: imgBytes.toString("base64"),
          },
        });
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
        }
      );

      // Extract images from response
      const candidate = data.candidates?.[0];
      if (!candidate?.content?.parts) {
        return { content: [{ type: "text", text: "No content returned. The prompt may have been blocked by safety filters." }] };
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
        const msg = textParts.length > 0
          ? `No images generated. Model response: ${textParts.join(" ")}`
          : "No images returned. Try rephrasing the prompt.";
        return { content: [{ type: "text", text: msg }] };
      }

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
  "Generate a video using Google Veo 3.1 with native audio. Saves MP4 to ~/Desktop/gemini-outputs/. Generation takes 11s to 6 minutes.",
  {
    prompt: z.string().describe("Text description of the video to generate"),
    durationSeconds: z
      .enum(["4", "6", "8"])
      .default("6")
      .describe("Video duration: 4, 6, or 8 seconds"),
    aspectRatio: z
      .enum(["9:16", "16:9"])
      .default("16:9")
      .describe("Aspect ratio of the generated video"),
    resolution: z
      .enum(["720p", "1080p", "4k"])
      .default("720p")
      .describe("Video resolution (1080p and 4k require 8s duration)"),
  },
  async ({ prompt, durationSeconds, aspectRatio, resolution }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    const duration = parseInt(durationSeconds, 10);

    // 1080p and 4k require 8s duration
    if ((resolution === "1080p" || resolution === "4k") && duration !== 8) {
      return {
        content: [{ type: "text", text: `Error: ${resolution} resolution requires 8 second duration.` }],
      };
    }

    try {
      // Step 1: Submit generation request
      const operation = await geminiRequest(
        "/models/veo-3.1-generate-preview:predictLongRunning",
        {
          instances: [{ prompt }],
          parameters: {
            numberOfVideos: 1,
            durationSeconds: duration,
            aspectRatio,
            resolution,
            personGeneration: "dont_allow",
          },
        }
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

        // Download video file
        const downloadUrl = `${videoUri}?alt=media&key=${GEMINI_API_KEY}`;
        const videoRes = await fetch(downloadUrl);
        if (!videoRes.ok) {
          throw new Error(`Failed to download video: ${videoRes.status}`);
        }

        const videoBuffer = Buffer.from(await videoRes.arrayBuffer());
        const filename = `veo3-${timestamp()}-${i + 1}.mp4`;
        const filepath = path.join(outputDir, filename);
        fs.writeFileSync(filepath, videoBuffer);
        savedFiles.push(filepath);
      }

      return {
        content: [
          {
            type: "text",
            text: `Generated ${savedFiles.length} video(s):\n${savedFiles.join("\n")}`,
          },
        ],
      };
    } catch (err) {
      if (err.message.includes("403") || err.message.includes("404")) {
        return {
          content: [
            {
              type: "text",
              text: `Veo 3.1 access error: ${err.message}\n\nThis likely means Veo 3.1 is not enabled for your API key. Check that your Gemini API plan supports video generation.`,
            },
          ],
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

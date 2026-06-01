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
  "veo-3.1-generate-preview": { perSecond: 0.40 },                 // quality, with audio
  "veo-3.1-fast-generate-preview": { perSecond: 0.15 },            // fast, with audio
  "veo-3.1-lite-generate-preview": { perSecond: 0.08 },            // lite (estimate; verify)
};

const VEO_MODELS = {
  quality: "veo-3.1-generate-preview",
  fast: "veo-3.1-fast-generate-preview",
  lite: "veo-3.1-lite-generate-preview",
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
  const resolved = path.resolve(filePath.replace(/^~/, os.homedir()));
  if (!fs.existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }
  let bytes;
  try {
    bytes = fs.readFileSync(resolved);
  } catch (err) {
    if (err.code === "EPERM" || err.code === "EACCES") {
      // macOS may restrict access via extended attributes (com.apple.macl)
      // even when the file is owner-readable. Check for ACL attributes.
      throw new Error(
        `Permission denied reading ${resolved}. ` +
        `This is usually a macOS privacy restriction (com.apple.macl extended attribute) ` +
        `on a parent directory, not a file-permission issue. ` +
        `To fix: (1) remove the attribute with \`xattr -rd com.apple.macl <parent-dir>\`, ` +
        `(2) grant Full Disk Access to your terminal/Node binary in System Settings → Privacy & Security → Full Disk Access, ` +
        `or (3) copy the file to ~/Desktop/gemini-outputs/ and pass that path instead. ` +
        `Original error: ${err.message}`
      );
    }
    throw err;
  }
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
  "Generate images using Google Imagen 4 (text-to-image only). For reference-image-guided generation, use gemini_native_image instead. Saves PNG to ~/Desktop/gemini-outputs/.",
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
  async ({ prompt, aspectRatio, sampleCount, imageSize, personGeneration, seed }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      const instance = { prompt };

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
        params: { sampleCount, aspectRatio, imageSize, personGeneration },
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
        // Gemini API returns camelCase (inlineData), not snake_case (inline_data)
        const imageData = part.inline_data || part.inlineData;
        if (imageData) {
          const mimeType = imageData.mime_type || imageData.mimeType || "image/png";
          const b64 = imageData.data;
          if (b64) {
            const buffer = Buffer.from(b64, "base64");
            const ext = mimeType === "image/jpeg" ? "jpg" : "png";
            const filename = `nanob-${timestamp()}-${savedFiles.length + 1}.${ext}`;
            const filepath = path.join(outputDir, filename);
            fs.writeFileSync(filepath, buffer);
            savedFiles.push(filepath);
          }
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
  "Generate a video using Google Veo 3.1 with native audio. Supports text-to-video, image-to-video (start frame), first+last frame interpolation (seamless loops), reference-image character/style consistency (ingredients-to-video, up to 3), and extension of a previous Veo-generated video URI. Saves MP4 to ~/Desktop/gemini-outputs/ unless outputDir is set.",
  {
    prompt: z.string().describe("Text description of the video to generate"),
    model: z
      .enum(["quality", "fast", "lite"])
      .default("quality")
      .describe("Veo 3.1 variant: quality (best), fast (cheaper, faster), lite (cheapest, no reference/extension)"),
    durationSeconds: z
      .enum(["4", "6", "8"])
      .default("6")
      .describe("Video duration: 4, 6, or 8 seconds. Forced to 8 when using 1080p/4k, referenceImages, or extension."),
    aspectRatio: z
      .enum(["9:16", "16:9"])
      .default("16:9")
      .describe("Aspect ratio"),
    resolution: z
      .enum(["720p", "1080p", "4k"])
      .default("720p")
      .describe("Video resolution (1080p and 4k require 8s duration; extension forces 720p)"),
    inputImagePath: z
      .string()
      .optional()
      .describe("Optional: path to a start-frame image for image-to-video animation"),
    lastFramePath: z
      .string()
      .optional()
      .describe("Optional: path to an end-frame image. Requires inputImagePath. Use same image for both to make a seamless loop, and include 'a seamless loop' in the prompt."),
    referenceImagePaths: z
      .array(z.string())
      .max(3)
      .optional()
      .describe("Optional: up to 3 reference images for character/object/scene consistency (ingredients-to-video). Not supported on the 'lite' model. Forces durationSeconds=8."),
    extendFromVideoUri: z
      .string()
      .optional()
      .describe("Optional: URI of a previously-generated Veo video to extend (returned from a prior call; valid for 2 days). Forces 720p and durationSeconds=8. Not supported on the 'lite' model."),
    personGeneration: z
      .enum(["dont_allow", "allow_adult", "allow_all"])
      .default("allow_adult")
      .describe("Person/face generation policy. Forced to allow_adult for image-to-video, interpolation, and EU/UK/CH/MENA regions."),
    outputDir: z
      .string()
      .optional()
      .describe("Optional: absolute path to write MP4(s) to. Defaults to ~/Desktop/gemini-outputs/."),
  },
  async ({ prompt, model, durationSeconds, aspectRatio, resolution, inputImagePath, lastFramePath, referenceImagePaths, extendFromVideoUri, personGeneration, outputDir }, extra) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    const modelId = VEO_MODELS[model];
    const usingReferences = referenceImagePaths && referenceImagePaths.length > 0;
    const usingExtension = !!extendFromVideoUri;
    const usingLastFrame = !!lastFramePath;

    // ── Constraint validation ────────────────────────────────
    if (usingLastFrame && !inputImagePath) {
      return { content: [{ type: "text", text: "Error: lastFramePath requires inputImagePath (first/last frame interpolation needs both ends)." }] };
    }
    if (model === "lite" && (usingReferences || usingExtension)) {
      return { content: [{ type: "text", text: "Error: referenceImages and extension are not supported on the 'lite' model. Use 'quality' or 'fast'." }] };
    }
    if (usingExtension && resolution !== "720p") {
      return { content: [{ type: "text", text: "Error: video extension requires resolution='720p'." }] };
    }

    // Forced-to-8s combos
    let duration = parseInt(durationSeconds, 10);
    const force8Reasons = [];
    if (resolution === "1080p" || resolution === "4k") force8Reasons.push(resolution);
    if (usingReferences) force8Reasons.push("referenceImages");
    if (usingExtension) force8Reasons.push("extension");
    if (force8Reasons.length > 0 && duration !== 8) {
      return {
        content: [{ type: "text", text: `Error: ${force8Reasons.join(", ")} require durationSeconds=8 (got ${duration}).` }],
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

      // First/last frame interpolation
      if (usingLastFrame) {
        const img = readImageAsBase64(lastFramePath);
        instance.lastFrame = {
          bytesBase64Encoded: img.data,
          mimeType: img.mimeType,
        };
      }

      // Ingredients-to-video (reference images for consistency)
      if (usingReferences) {
        instance.referenceImages = referenceImagePaths.map((p) => {
          const img = readImageAsBase64(p);
          return {
            image: { bytesBase64Encoded: img.data, mimeType: img.mimeType },
            referenceType: "asset",
          };
        });
      }

      // Scene extension (continue a prior Veo video)
      if (usingExtension) {
        instance.video = { uri: extendFromVideoUri, mimeType: "video/mp4" };
      }

      // Build parameters
      const parameters = {
        durationSeconds: duration,
        aspectRatio,
        resolution,
        personGeneration,
      };

      // Step 1: Submit generation request
      const operation = await geminiRequest(
        `/models/${modelId}:predictLongRunning`,
        { instances: [instance], parameters }
      );

      if (!operation.name) {
        return { content: [{ type: "text", text: `Unexpected response: ${JSON.stringify(operation)}` }] };
      }

      // Step 2: Poll for completion (max 7 minutes)
      // Send progress notifications to keep the MCP client connection alive
      const maxWaitMs = 7 * 60 * 1000;
      const pollIntervalMs = 5000;
      const startTime = Date.now();
      let result;
      let pollCount = 0;

      const progressToken = extra?._meta?.progressToken;

      while (Date.now() - startTime < maxWaitMs) {
        await new Promise((r) => setTimeout(r, pollIntervalMs));
        result = await geminiGet(operation.name);
        pollCount++;

        // Send progress notification to prevent client timeout
        try {
          const elapsed = Math.round((Date.now() - startTime) / 1000);
          if (progressToken !== undefined) {
            await extra.sendNotification({
              method: "notifications/progress",
              params: {
                progressToken,
                progress: Math.min(pollCount * 5, 95),
                total: 100,
                message: `Generating video... (${elapsed}s)`,
              },
            });
          }
          // Also send a log notification as a keep-alive signal
          await extra.sendNotification({
            method: "notifications/message",
            params: {
              level: "info",
              data: `Video generation in progress... (${elapsed}s)`,
            },
          });
        } catch {
          // Don't fail if notification sending fails
        }

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

      const resolvedOutputDir = outputDir
        ? (fs.mkdirSync(path.resolve(outputDir.replace(/^~/, os.homedir())), { recursive: true }), path.resolve(outputDir.replace(/^~/, os.homedir())))
        : getOutputDir();
      const savedFiles = [];
      const videoUris = [];

      for (let i = 0; i < samples.length; i++) {
        const videoUri = samples[i].video?.uri;
        if (!videoUri) continue;
        videoUris.push(videoUri);

        const videoRes = await fetch(videoUri, {
          headers: { "x-goog-api-key": GEMINI_API_KEY },
          redirect: "follow",
        });
        if (!videoRes.ok) {
          throw new Error(`Failed to download video: ${videoRes.status}`);
        }

        const videoBuffer = Buffer.from(await videoRes.arrayBuffer());
        const filename = `veo3-${model}-${timestamp()}-${i + 1}.mp4`;
        const filepath = path.join(resolvedOutputDir, filename);
        fs.writeFileSync(filepath, videoBuffer);
        savedFiles.push(filepath);
      }

      // Log usage
      const veoPricePerSec = PRICING[modelId]?.perSecond ?? 0.15;
      logUsage({
        tool: "gemini_generate_video",
        model: modelId,
        params: {
          durationSeconds: duration,
          aspectRatio,
          resolution,
          personGeneration,
          hasStartFrame: !!inputImagePath,
          hasLastFrame: usingLastFrame,
          referenceImageCount: usingReferences ? referenceImagePaths.length : 0,
          isExtension: usingExtension,
        },
        outputCount: savedFiles.length,
        estimatedCost: savedFiles.length * duration * veoPricePerSec,
      });

      const uriHint = videoUris.length > 0
        ? `\n\nVideo URIs (valid ~2 days, pass to extendFromVideoUri to extend):\n${videoUris.join("\n")}`
        : "";

      return {
        content: [{ type: "text", text: `Generated ${savedFiles.length} video(s) with ${modelId}:\n${savedFiles.join("\n")}${uriHint}` }],
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

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
  version: "1.0.0",
});

// ── Tool: Generate Image (Imagen 3) ────────────────────────

server.tool(
  "gemini_generate_image",
  "Generate an image using Google Imagen 3. Saves PNG to ~/Desktop/gemini-outputs/.",
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
  },
  async ({ prompt, aspectRatio, sampleCount }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      const data = await geminiRequest(
        "/models/imagen-3.0-generate-002:predict",
        {
          instances: [{ prompt }],
          parameters: {
            sampleCount,
            aspectRatio,
          },
        }
      );

      if (!data.predictions || data.predictions.length === 0) {
        return {
          content: [{ type: "text", text: "No images returned. The prompt may have been blocked by safety filters." }],
        };
      }

      const outputDir = getOutputDir();
      const savedFiles = [];

      for (let i = 0; i < data.predictions.length; i++) {
        const prediction = data.predictions[i];
        const buffer = Buffer.from(prediction.bytesBase64Encoded, "base64");
        const filename = `imagen-${timestamp()}-${i + 1}.png`;
        const filepath = path.join(outputDir, filename);
        fs.writeFileSync(filepath, buffer);
        savedFiles.push(filepath);
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

// ── Tool: Generate Video (Veo 2) ───────────────────────────

server.tool(
  "gemini_generate_video",
  "Generate a video using Google Veo 2. Saves MP4 to ~/Desktop/gemini-outputs/. Video generation takes 1-3 minutes.",
  {
    prompt: z.string().describe("Text description of the video to generate"),
    durationSeconds: z
      .number()
      .int()
      .min(5)
      .max(8)
      .default(5)
      .describe("Video duration in seconds (5 or 8)"),
    aspectRatio: z
      .enum(["9:16", "16:9"])
      .default("16:9")
      .describe("Aspect ratio of the generated video"),
  },
  async ({ prompt, durationSeconds, aspectRatio }) => {
    if (!GEMINI_API_KEY) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    try {
      // Step 1: Submit generation request
      const operation = await geminiRequest(
        "/models/veo-2.0-generate-001:predictLongRunning",
        {
          instances: [{ prompt }],
          parameters: {
            sampleCount: 1,
            durationSeconds,
            aspectRatio,
            personGeneration: "dont_allow",
          },
        }
      );

      if (!operation.name) {
        return { content: [{ type: "text", text: `Unexpected response: ${JSON.stringify(operation)}` }] };
      }

      // Step 2: Poll for completion (max 5 minutes)
      const maxWaitMs = 5 * 60 * 1000;
      const pollIntervalMs = 5000;
      const startTime = Date.now();
      let result;

      while (Date.now() - startTime < maxWaitMs) {
        await new Promise((r) => setTimeout(r, pollIntervalMs));
        result = await geminiGet(operation.name);

        if (result.done) break;
      }

      if (!result?.done) {
        return { content: [{ type: "text", text: "Video generation timed out after 5 minutes. Try again or use a simpler prompt." }] };
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
        const filename = `veo2-${timestamp()}-${i + 1}.mp4`;
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
              text: `Veo 2 access error: ${err.message}\n\nThis likely means Veo 2 is not enabled for your API key. Check that your Gemini API plan supports video generation.`,
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

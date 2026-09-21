import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import os from "os";

// ── API key pool ────────────────────────────────────────────
// Backward compatible: a single GEMINI_API_KEY still works. Additionally
// supports a comma-list (GEMINI_API_KEYS) and/or a keys file
// (GEMINI_API_KEYS_FILE, one key per line, # comments allowed). On a 429 /
// RESOURCE_EXHAUSTED during video submit, the pool rotates to the next key.
function loadKeys() {
  const out = [];
  const file = process.env.GEMINI_API_KEYS_FILE;
  if (file) {
    try {
      const txt = fs.readFileSync(file.replace(/^~/, os.homedir()), "utf8");
      for (const line of txt.split("\n")) {
        const k = line.trim();
        if (k && !k.startsWith("#")) out.push(k);
      }
    } catch { /* missing/unreadable keys file → ignore */ }
  }
  if (process.env.GEMINI_API_KEYS) {
    for (const k of process.env.GEMINI_API_KEYS.split(",")) {
      const t = k.trim();
      if (t) out.push(t);
    }
  }
  if (process.env.GEMINI_API_KEY) out.push(process.env.GEMINI_API_KEY.trim());
  return [...new Set(out)].filter(Boolean);
}
const API_KEYS = loadKeys();
let keyIdx = 0;
function activeKey() { return API_KEYS[keyIdx]; }
function rotateKey() { keyIdx = (keyIdx + 1) % Math.max(API_KEYS.length, 1); return keyIdx; }
function hasKeys() { return API_KEYS.length > 0; }

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

// ── Usage Logging ───────────────────────────────────────────

const USAGE_LOG_PATH = path.join(os.homedir(), "Desktop", "gemini-outputs", "usage-log.jsonl");

// Image models, chosen per call via the `model` param (flash | pro).
//   flash = gemini-3.1-flash-image (Nano Banana): fast, cheap, ~95% of Pro
//           quality, image-to-image chaining for consistency. The default.
//   pro   = gemini-3-pro-image (Nano Banana Pro): top fidelity and BY FAR the
//           best in-image TEXT rendering (logos, signage, posters), brand /
//           pixel accuracy; ~2x cost, ~4x slower. Use for hero/final/text work.
// Both are called via :generateContent with the same imageConfig shape.
// Imagen 4 (imagen-4.0-*) is discontinued by Google on 2026-08-17 (404 after).
// See: ai.google.dev/gemini-api/docs/image-generation
const IMAGE_MODELS = {
  flash: "gemini-3.1-flash-image",
  pro: "gemini-3-pro-image",
  lite: "gemini-3.1-flash-lite-image", // cheapest; 1K only; weak at multi-reference / sequential edits
};
const DEFAULT_IMAGE_MODEL = "flash";

// Pricing table (USD per output image) — update if Google changes rates.
// Source: ai.google.dev/gemini-api/docs/pricing (verified 2026-06-15).
const PRICING = {
  "gemini-3.1-flash-image": { "512": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151 }, // Nano Banana (Flash)
  "gemini-3-pro-image":     { "512": 0.134, "1K": 0.134, "2K": 0.134, "4K": 0.24 },  // Nano Banana Pro
  "gemini-3.1-flash-lite-image": { "1K": 0.0336 },                 // Nano Banana 2 Lite
  // Veo 3.1 (USD per second, with audio), by resolution. Verified 2026-09-21.
  "veo-3.1-generate-preview":      { perSecond: { "720p": 0.40, "1080p": 0.40, "4k": 0.60 } },
  "veo-3.1-fast-generate-preview": { perSecond: { "720p": 0.10, "1080p": 0.12, "4k": 0.30 } },
  "veo-3.1-lite-generate-preview": { perSecond: { "720p": 0.05, "1080p": 0.08 } }, // no 4k
};

const VEO_MODELS = {
  quality: "veo-3.1-generate-preview",
  fast: "veo-3.1-fast-generate-preview",
  lite: "veo-3.1-lite-generate-preview",
  // Veo 3.0 (veo-3.0-*-001) is gone from Google's model list (404); removed 2026-09-21.
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

// Resolve the target dir for a generation: base is `absOverride` (an explicit absolute
// outputDir) or ~/Desktop/gemini-outputs, optionally nested under a sanitized per-project/
// purpose subfolder (nested paths ok; `..`/absolute segments stripped). Created if missing.
function resolveOutDir(project, absOverride) {
  const base = absOverride
    ? path.resolve(absOverride.replace(/^~/, os.homedir()))
    : getOutputDir();
  const safe = String(project || "")
    .split("/")
    .map((s) => s.trim().replace(/[^A-Za-z0-9._-]/g, "-"))
    .filter((s) => s && s !== "." && s !== "..")
    .join("/");
  const dir = safe ? path.join(base, safe) : base;
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// Write a plain-text sidecar next to a saved artifact: SAME basename, `.txt` extension.
// Holds the exact prompt (the only reproduction handle — images have no seed) plus a
// compact settings footer. Best-effort: never throws back into the caller.
function writeSidecar(filepath, prompt, meta = {}) {
  try {
    const side = filepath.replace(/\.[^./]+$/, "") + ".txt";
    const metaLines = Object.entries(meta)
      .filter(([, v]) => v !== undefined && v !== null && v !== "" && !(Array.isArray(v) && v.length === 0))
      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`);
    const body = [prompt || ""]
      .concat(metaLines.length ? ["", "----", ...metaLines] : [])
      .join("\n") + "\n";
    fs.writeFileSync(side, body);
    return side;
  } catch {
    return null;
  }
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

async function geminiRequest(endpoint, body, key = activeKey()) {
  const url = `${BASE_URL}${endpoint}?key=${key}`;
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

// Pooled variant used by the image handlers (Imagen 4 + Nano Banana), which
// otherwise have no failover. Tries the active key; on 429 / RESOURCE_EXHAUSTED
// / quota / spend-cap it rotates to the next pool key and retries (once per
// key), leaving keyIdx on the working key for subsequent calls. Mirrors the
// video handler's rotation so a capped key can't silently block images.
async function geminiRequestPooled(endpoint, body) {
  let lastErr;
  for (let attempt = 0; attempt < Math.max(API_KEYS.length, 1); attempt++) {
    try {
      return await geminiRequest(endpoint, body, activeKey());
    } catch (e) {
      lastErr = e;
      if (/429|RESOURCE_EXHAUSTED|quota|spend/i.test(e.message) && API_KEYS.length > 1) {
        rotateKey();
        continue;
      }
      throw e;
    }
  }
  throw new Error(
    `All ${API_KEYS.length} pooled API key(s) exhausted (429 / quota / monthly spend-cap). ` +
    `Raise a spend cap at https://ai.studio/spend, or add another key to GEMINI_API_KEYS (sops secrets.enc.env) and restart. ` +
    `Last error: ${lastErr?.message || "unknown"}`
  );
}

async function geminiGet(endpoint, key = activeKey()) {
  const url = `${BASE_URL}/${endpoint}?key=${key}`;
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
  version: "2.2.0",
});

// ── Tool: Generate Image (Gemini 3.1 Flash Image, text-to-image) ──
// Migrated 2026-06-15 off the deprecated Imagen 4 predict endpoint
// (imagen-4.0-generate-001, removed by Google 2026-08-17) to
// gemini-3.1-flash-image via :generateContent. generateContent returns one
// image per call, so sampleCount loops the request. Imagen-only parameters
// (seed / addWatermark) have no equivalent here and were dropped.

server.tool(
  "gemini_generate_image",
  "Generate images from text using Google's Nano Banana image models (text-to-image). Produces 1-4 images per call (each a separate request). model='flash' (gemini-3.1-flash-image, default) is fast/cheap; model='pro' (gemini-3-pro-image) is the quality tier with far better in-image TEXT rendering (logos, signage, posters) — use it for hero/final/text work. For reference-image-guided generation or editing an existing image, use gemini_native_image instead. Saves PNG/JPG to ~/Desktop/gemini-outputs/ (pass 'project' to nest outputs in a per-project/purpose subfolder; each image also gets a same-named .txt sidecar with the prompt + settings for reproduction).",
  {
    prompt: z.string().describe("Text description of the image to generate"),
    model: z
      .enum(["flash", "pro", "lite"])
      .default("flash")
      .describe("flash = gemini-3.1-flash-image (fast, cheap, default); pro = gemini-3-pro-image (best fidelity + in-image text, ~2x cost, ~4x slower); lite = gemini-3.1-flash-lite-image (cheapest ~$0.034, 1K only, weak at reference-heavy edits; for bulk drafts)."),
    aspectRatio: z
      .enum(["1:1", "1:4", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])
      .default("1:1")
      .describe("Aspect ratio"),
    sampleCount: z
      .number()
      .int()
      .min(1)
      .max(4)
      .default(1)
      .describe("Number of images to generate (1-4). Each is a separate API call."),
    imageSize: z
      .enum(["512", "1K", "2K", "4K"])
      .default("1K")
      .describe("Output resolution: 512 (0.5K), 1K, 2K, or 4K"),
    thinkingLevel: z
      .enum(["minimal", "high"])
      .optional()
      .describe("Optional reasoning depth for the 3.1 models (flash, lite): 'high' plans composition before drawing (better on complex prompts, slower). Ignored for pro. Omit for Google's default (minimal)."),
    personGeneration: z
      .enum(["dont_allow", "allow_adult", "allow_all"])
      .default("allow_adult")
      .describe("Person/face generation policy"),
    project: z
      .string()
      .optional()
      .describe("Optional subfolder under ~/Desktop/gemini-outputs to group outputs by project/purpose (e.g. 'odyssey-retold/heroes'). Nested paths allowed; sanitized; created if missing. Omit to save in the root."),
  },
  async ({ prompt, model, aspectRatio, sampleCount, imageSize, thinkingLevel, personGeneration, project }) => {
    if (!hasKeys()) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    const modelId = IMAGE_MODELS[model] || IMAGE_MODELS[DEFAULT_IMAGE_MODEL];
    if (model === "lite" && imageSize !== "1K") {
      return { content: [{ type: "text", text: "Error: model='lite' supports imageSize '1K' only." }] };
    }
    const genConfigExtra = thinkingLevel && model !== "pro" ? { thinkingConfig: { thinkingLevel } } : {};

    try {
      const outputDir = resolveOutDir(project);
      const savedFiles = [];
      const textParts = [];
      const errors = [];

      for (let n = 0; n < sampleCount; n++) {
        try {
          const data = await geminiRequestPooled(
            `/models/${modelId}:generateContent`,
            {
              contents: [{ role: "user", parts: [{ text: prompt }] }],
              generationConfig: {
                responseModalities: ["TEXT", "IMAGE"],
                imageConfig: { aspectRatio, imageSize },
                ...genConfigExtra,
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
          const parts = candidate?.content?.parts || [];
          let gotImage = false;
          for (const part of parts) {
            if (part.text) textParts.push(part.text);
            // Gemini API returns camelCase (inlineData); accept snake_case too.
            const imageData = part.inline_data || part.inlineData;
            if (imageData?.data) {
              const mimeType = imageData.mime_type || imageData.mimeType || "image/png";
              const buffer = Buffer.from(imageData.data, "base64");
              const ext = mimeType === "image/jpeg" ? "jpg" : "png";
              const filename = `gemimg-${timestamp()}-${savedFiles.length + 1}.${ext}`;
              const filepath = path.join(outputDir, filename);
              fs.writeFileSync(filepath, buffer);
              savedFiles.push(filepath);
              writeSidecar(filepath, prompt, { tool: "gemini_generate_image", model: modelId, aspectRatio, imageSize, thinkingLevel });
              gotImage = true;
            }
          }
          if (!gotImage) {
            errors.push(`sample ${n + 1}: no image (finishReason: ${candidate?.finishReason}, blockReason: ${data?.promptFeedback?.blockReason})`);
          }
        } catch (e) {
          errors.push(`sample ${n + 1}: ${e.message}`);
        }
      }

      if (savedFiles.length === 0) {
        const detail = errors.length ? `\n${errors.join("\n")}` : "";
        const note = textParts.length ? ` Model response: ${textParts.join(" ")}` : "";
        return { content: [{ type: "text", text: `No images returned. The prompt may have been blocked by safety filters.${note}${detail}` }] };
      }

      // Log usage
      const pricePerImage = PRICING[modelId]?.[imageSize] ?? 0.04;
      logUsage({
        tool: "gemini_generate_image",
        model: modelId,
        params: { sampleCount, aspectRatio, imageSize, personGeneration },
        outputCount: savedFiles.length,
        estimatedCost: savedFiles.length * pricePerImage,
      });

      let result = `Generated ${savedFiles.length} image(s):\n${savedFiles.join("\n")}`;
      if (errors.length) result += `\n\n${errors.length} of ${sampleCount} did not return an image:\n${errors.join("\n")}`;
      if (textParts.length) result += `\n\nModel notes: ${textParts.join(" ")}`;
      return { content: [{ type: "text", text: result }] };
    } catch (err) {
      return { content: [{ type: "text", text: `Error: ${err.message}` }] };
    }
  }
);

// ── Tool: Native Image Gen/Edit (Nano Banana) ──────────────

server.tool(
  "gemini_native_image",
  "Generate or edit images using Google's Nano Banana image models. Supports multiple input images, image editing, multi-turn style, and up to 4K — this is the consistency engine: chain from an approved reference via inputImagePaths to keep a face/place/object identical. model='flash' (gemini-3.1-flash-image, default) is the fast workhorse for chaining/bulk; model='pro' (gemini-3-pro-image) is the quality tier with far better in-image TEXT rendering — use for hero/final/text work. Saves PNG/JPG to ~/Desktop/gemini-outputs/ (pass 'project' to nest outputs in a per-project/purpose subfolder; each image also gets a same-named .txt sidecar with the prompt + settings for reproduction).",
  {
    prompt: z.string().describe("Text description or editing instruction"),
    model: z
      .enum(["flash", "pro", "lite"])
      .default("flash")
      .describe("flash = gemini-3.1-flash-image (fast, cheap, default); pro = gemini-3-pro-image (best fidelity + in-image text, ~2x cost, ~4x slower); lite = gemini-3.1-flash-lite-image (cheapest ~$0.034, 1K only, weak at reference-heavy edits; for bulk drafts)."),
    aspectRatio: z
      .enum(["1:1", "1:4", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])
      .default("1:1")
      .describe("Aspect ratio (14 options)"),
    imageSize: z
      .enum(["512", "1K", "2K", "4K"])
      .default("1K")
      .describe("Output resolution"),
    thinkingLevel: z
      .enum(["minimal", "high"])
      .optional()
      .describe("Optional reasoning depth for the 3.1 models (flash, lite): 'high' plans composition before drawing (better on complex prompts, slower). Ignored for pro. Omit for Google's default (minimal)."),
    inputImagePaths: z
      .array(z.string())
      .optional()
      .describe("Optional: array of absolute paths to input images for editing, reference, or multi-image context"),
    personGeneration: z
      .enum(["dont_allow", "allow_adult", "allow_all"])
      .default("allow_adult")
      .describe("Person/face generation policy"),
    project: z
      .string()
      .optional()
      .describe("Optional subfolder under ~/Desktop/gemini-outputs to group outputs by project/purpose (e.g. 'odyssey-retold/heroes'). Nested paths allowed; sanitized; created if missing. Omit to save in the root."),
  },
  async ({ prompt, model, aspectRatio, imageSize, thinkingLevel, inputImagePaths, personGeneration, project }) => {
    if (!hasKeys()) {
      return { content: [{ type: "text", text: "Error: GEMINI_API_KEY environment variable is not set." }] };
    }

    const modelId = IMAGE_MODELS[model] || IMAGE_MODELS[DEFAULT_IMAGE_MODEL];
    if (model === "lite" && imageSize !== "1K") {
      return { content: [{ type: "text", text: "Error: model='lite' supports imageSize '1K' only." }] };
    }
    const genConfigExtra = thinkingLevel && model !== "pro" ? { thinkingConfig: { thinkingLevel } } : {};

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

      const data = await geminiRequestPooled(
        `/models/${modelId}:generateContent`,
        {
          contents: [{ role: "user", parts }],
          generationConfig: {
            responseModalities: ["TEXT", "IMAGE"],
            imageConfig: {
              aspectRatio,
              imageSize,
            },
            ...genConfigExtra,
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

      const outputDir = resolveOutDir(project);
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
            writeSidecar(filepath, prompt, { tool: "gemini_native_image", model: modelId, aspectRatio, imageSize, thinkingLevel, refs: inputImagePaths });
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
      const nanoPricePerImage = PRICING[modelId]?.[imageSize] ?? 0.04;
      logUsage({
        tool: "gemini_native_image",
        model: modelId,
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
  "Generate a video using Google Veo 3.1 with native audio. Supports text-to-video, image-to-video (start frame), first+last frame interpolation (seamless loops), reference-image character/style consistency (ingredients-to-video, up to 3), and extension of a previous Veo-generated video URI. Saves MP4 to ~/Desktop/gemini-outputs/ unless outputDir is set (pass 'project' to nest in a per-project/purpose subfolder; each MP4 also gets a same-named .txt sidecar with the prompt + settings). KEY POOL & FAILOVER: this server loads an API-key pool (GEMINI_API_KEYS comma-list, then GEMINI_API_KEY). It uses key 0 by default and AUTOMATICALLY rotates to the next key when one returns 429 / quota / a monthly spend-cap. To force a specific key pass keyIndex (0-based). If a call returns 'All N pooled keys exhausted', every key is blocked — raise a spend cap at https://ai.studio/spend or add another key to GEMINI_API_KEYS.",
  {
    prompt: z.string().describe("Text description of the video to generate"),
    negativePrompt: z
      .string()
      .optional()
      .describe("Negative prompt (comma-separated nouns) Veo will suppress, e.g. 'camera motion, pan, tilt, zoom, dolly, blur, text, people, boats'. Not supported on 'lite' (Google 400s)."),
    seed: z
      .number()
      .int()
      .min(0)
      .optional()
      .describe("Optional integer seed for more repeatable output (Veo 3.x; not fully deterministic). Note it in the sidecar to reproduce."),
    keyIndex: z
      .number()
      .int()
      .min(0)
      .optional()
      .describe("Force a specific 0-based API key from the pool (skips 429 auto-rotation). Omit for normal behaviour."),
    model: z
      .enum(["quality", "fast", "lite"])
      .default("quality")
      .describe("Veo 3.1 variant: quality (best, 4k ok), fast (4k ok), lite (cheapest; no 4k, no negativePrompt, no referenceImages, no extension)."),
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
      .default("allow_all")
      .describe("Person/face generation policy. Default 'allow_all' — as of 2026-06 the Gemini API Veo endpoint REJECTS 'allow_adult' and 'dont_allow' with a 400 'currently not supported'; only 'allow_all' is accepted. Keep 'allow_all' unless Google re-enables the others."),
    outputDir: z
      .string()
      .optional()
      .describe("Optional: absolute path to write MP4(s) to. Defaults to ~/Desktop/gemini-outputs/."),
    project: z
      .string()
      .optional()
      .describe("Optional subfolder to group outputs by project/purpose (e.g. 'odyssey-retold/broll'), nested under outputDir or ~/Desktop/gemini-outputs. Sanitized; created if missing."),
  },
  async ({ prompt, negativePrompt, seed, keyIndex, model, durationSeconds, aspectRatio, resolution, inputImagePath, lastFramePath, referenceImagePaths, extendFromVideoUri, personGeneration, outputDir, project }, extra) => {
    if (!hasKeys()) {
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
    if (model === "lite" && resolution === "4k") {
      return { content: [{ type: "text", text: "Error: the 'lite' model does not support 4k. Use 'quality' or 'fast', or resolution 720p/1080p." }] };
    }
    if (model === "lite" && negativePrompt) {
      return { content: [{ type: "text", text: "Error: negativePrompt is not supported on the 'lite' model. Use 'quality' or 'fast', or fold the exclusions into the prompt." }] };
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
      };
      if (personGeneration) parameters.personGeneration = personGeneration;
      if (negativePrompt) parameters.negativePrompt = negativePrompt;
      if (seed !== undefined) parameters.seed = seed;

      // Step 1: Submit generation request.
      // If keyIndex is given, pin to that pool key (no rotation); otherwise
      // use the active key and rotate to the next on 429 / quota.
      const pinned = Number.isInteger(keyIndex) && keyIndex >= 0 && keyIndex < API_KEYS.length;
      let operation, vidKey, lastErr;
      const attempts = pinned ? 1 : Math.max(API_KEYS.length, 1);
      for (let a = 0; a < attempts; a++) {
        vidKey = pinned ? API_KEYS[keyIndex] : activeKey();
        try {
          operation = await geminiRequest(
            `/models/${modelId}:predictLongRunning`,
            { instances: [instance], parameters },
            vidKey
          );
          break;
        } catch (e) {
          lastErr = e;
          if (!pinned && /429|RESOURCE_EXHAUSTED/i.test(e.message) && API_KEYS.length > 1) {
            rotateKey();
            continue;
          }
          throw e;
        }
      }
      if (!operation) {
        const detail = lastErr ? ` Last error: ${lastErr.message}` : "";
        throw new Error(
          `All ${API_KEYS.length} pooled API key(s) exhausted (429 / quota / monthly spend-cap). ` +
          `Raise a spend cap at https://ai.studio/spend, or add another key to GEMINI_API_KEYS (sops secrets.enc.env) and restart.${detail}`
        );
      }

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
        result = await geminiGet(operation.name, vidKey);
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

      const resolvedOutputDir = resolveOutDir(project, outputDir);
      const savedFiles = [];
      const videoUris = [];

      for (let i = 0; i < samples.length; i++) {
        const videoUri = samples[i].video?.uri;
        if (!videoUri) continue;
        videoUris.push(videoUri);

        const videoRes = await fetch(videoUri, {
          headers: { "x-goog-api-key": vidKey },
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
        writeSidecar(filepath, prompt, { tool: "gemini_generate_video", model: modelId, aspectRatio, resolution, durationSeconds: duration, seed, negativePrompt, refs: referenceImagePaths, startFrame: inputImagePath, lastFrame: lastFramePath });
      }

      // Log usage
      const veoPricePerSec = PRICING[modelId]?.perSecond?.[resolution] ?? 0.40;
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

/**
 * Clicky Proxy Worker — Gemini Edition
 *
 * Proxies all requests to Vertex AI Gemini so the app never ships
 * with raw GCP credentials. The service-account key is stored as a
 * Cloudflare Worker secret and exchanged for a short-lived OAuth2
 * access token on every request.
 *
 * Routes:
 *   POST /chat        → Vertex AI generateContent (streaming, vision + text)
 *   POST /tts         → Vertex AI generateContent (audio output / TTS)
 *   POST /transcribe  → Vertex AI generateContent (audio input / STT)
 */

interface Env {
  GOOGLE_CLIENT_EMAIL: string;   // service account email
  GOOGLE_PRIVATE_KEY: string;    // PEM private key (literal \n for newlines)
  GOOGLE_PROJECT_ID: string;
  GEMINI_MODEL: string;          // e.g. gemini-3.1-flash-lite-preview
  GEMINI_TTS_MODEL: string;      // e.g. gemini-3.1-flash-tts-preview
  GEMINI_STT_MODEL: string;      // e.g. gemini-3-flash-preview
  GEMINI_TTS_VOICE: string;      // e.g. Kore
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    try {
      if (url.pathname === "/chat")       return await handleChat(request, env);
      if (url.pathname === "/tts")        return await handleTTS(request, env);
      if (url.pathname === "/transcribe") return await handleTranscribe(request, env);
    } catch (error) {
      console.error(`[${url.pathname}] Unhandled error:`, error);
      return new Response(JSON.stringify({ error: String(error) }), {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }

    return new Response("Not found", { status: 404 });
  },
};

// ── OAuth2 token exchange ──────────────────────────────────────────────────

/** Exchanges the service-account credentials for a short-lived Bearer token. */
async function getAccessToken(env: Env): Promise<string> {
  const jwt = await createServiceAccountJWT(env.GOOGLE_CLIENT_EMAIL, env.GOOGLE_PRIVATE_KEY);

  const response = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: jwt,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`OAuth2 token exchange failed (${response.status}): ${body}`);
  }

  const data = await response.json() as { access_token?: string };
  if (!data.access_token) throw new Error("OAuth2 response missing access_token");
  return data.access_token;
}

/** Builds a RS256-signed JWT for the Google OAuth2 token endpoint. */
async function createServiceAccountJWT(clientEmail: string, rawPrivateKey: string): Promise<string> {
  const header = { alg: "RS256", typ: "JWT" };
  const now = Math.floor(Date.now() / 1000);
  const claims = {
    iss: clientEmail,
    scope: "https://www.googleapis.com/auth/cloud-platform",
    aud: "https://oauth2.googleapis.com/token",
    iat: now,
    exp: now + 3600,
  };

  const encodedHeader  = b64url(JSON.stringify(header));
  const encodedPayload = b64url(JSON.stringify(claims));
  const signingInput   = `${encodedHeader}.${encodedPayload}`;

  const privateKey = await importPrivateKey(rawPrivateKey);
  const signature  = await crypto.subtle.sign(
    { name: "RSASSA-PKCS1-v1_5" },
    privateKey,
    new TextEncoder().encode(signingInput)
  );

  return `${signingInput}.${arrayBufferToB64url(signature)}`;
}

/** Imports the PEM private key into the WebCrypto API. */
async function importPrivateKey(rawPem: string): Promise<CryptoKey> {
  // Expand literal \n sequences (stored as two chars in Wrangler secrets)
  const pem = rawPem.replace(/\\n/g, "\n");

  // Strip PEM headers/footers and whitespace to get the bare base64 body,
  // then drop any stray non-base64 chars (the key has corrupted \y \K bytes
  // that Go silently ignores — we do the same here).
  const base64 = pem
    .replace(/-----BEGIN PRIVATE KEY-----/g, "")
    .replace(/-----END PRIVATE KEY-----/g, "")
    .replace(/[^A-Za-z0-9+/=]/g, "");

  const der = Uint8Array.from(atob(base64), c => c.charCodeAt(0));

  return crypto.subtle.importKey(
    "pkcs8",
    der.buffer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"]
  );
}

// ── Vertex AI helpers ──────────────────────────────────────────────────────

function vertexEndpoint(projectId: string, model: string): string {
  return (
    `https://aiplatform.googleapis.com/v1/projects/${projectId}` +
    `/locations/global/publishers/google/models/${model}:generateContent`
  );
}

function vertexStreamingEndpoint(projectId: string, model: string): string {
  return (
    `https://aiplatform.googleapis.com/v1/projects/${projectId}` +
    `/locations/global/publishers/google/models/${model}:streamGenerateContent?alt=sse`
  );
}

async function vertexPost(
  endpoint: string,
  accessToken: string,
  body: unknown
): Promise<Response> {
  return fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

// ── Route handlers ─────────────────────────────────────────────────────────

/**
 * POST /chat
 * Body (JSON from Swift app):
 *   { systemPrompt, userPrompt, images: [{base64, mimeType, label}], history: [...], model? }
 *
 * Streams Vertex AI SSE back to the app unchanged.
 */
async function handleChat(request: Request, env: Env): Promise<Response> {
  const body = await request.json() as {
    systemPrompt?: string;
    userPrompt: string;
    images?: Array<{ base64: string; mimeType: string; label: string }>;
    history?: Array<{ userPlaceholder: string; assistantResponse: string }>;
    model?: string;
  };

  const model = body.model ?? env.GEMINI_MODEL ?? "gemini-3.1-flash-lite-preview";
  const accessToken = await getAccessToken(env);

  // Build Vertex AI contents array
  const contents: unknown[] = [];

  for (const turn of body.history ?? []) {
    contents.push({ role: "user",  parts: [{ text: turn.userPlaceholder }] });
    contents.push({ role: "model", parts: [{ text: turn.assistantResponse }] });
  }

  const currentParts: unknown[] = [];
  for (const image of body.images ?? []) {
    currentParts.push({ inlineData: { mimeType: image.mimeType, data: image.base64 } });
    currentParts.push({ text: image.label });
  }
  currentParts.push({ text: body.userPrompt });
  contents.push({ role: "user", parts: currentParts });

  const vertexBody = {
    system_instruction: { parts: [{ text: body.systemPrompt ?? "" }] },
    contents,
    generationConfig: { maxOutputTokens: 1024, responseModalities: ["TEXT"] },
  };

  const endpoint = vertexStreamingEndpoint(env.GOOGLE_PROJECT_ID, model);
  const upstream  = await vertexPost(endpoint, accessToken, vertexBody);

  if (!upstream.ok) {
    const errBody = await upstream.text();
    console.error(`[/chat] Vertex AI error ${upstream.status}: ${errBody}`);
    return new Response(errBody, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "text/event-stream",
      "cache-control": "no-cache",
    },
  });
}

/**
 * POST /tts
 * Body (JSON from Swift app): { text, voiceName? }
 * Returns: WAV audio bytes
 */
async function handleTTS(request: Request, env: Env): Promise<Response> {
  const body = await request.json() as { text: string; voiceName?: string };
  const voiceName    = body.voiceName ?? env.GEMINI_TTS_VOICE ?? "Kore";
  const ttsModel     = env.GEMINI_TTS_MODEL ?? "gemini-3.1-flash-tts-preview";
  const accessToken  = await getAccessToken(env);

  const vertexBody = {
    contents: [{ role: "user", parts: [{ text: body.text }] }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName },
        },
      },
    },
  };

  const endpoint = vertexEndpoint(env.GOOGLE_PROJECT_ID, ttsModel);
  const upstream  = await vertexPost(endpoint, accessToken, vertexBody);

  if (!upstream.ok) {
    const errBody = await upstream.text();
    console.error(`[/tts] Vertex AI error ${upstream.status}: ${errBody}`);
    return new Response(errBody, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  }

  const responseJson = await upstream.json() as {
    candidates?: Array<{
      content?: { parts?: Array<{ inlineData?: { data?: string; mimeType?: string } }> };
    }>;
  };

  const audioBase64 = responseJson
    ?.candidates?.[0]
    ?.content
    ?.parts?.[0]
    ?.inlineData
    ?.data;

  if (!audioBase64) {
    console.error("[/tts] No audio in Gemini response:", JSON.stringify(responseJson));
    return new Response(JSON.stringify({ error: "No audio in Gemini response" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }

  // Decode PCM base64 and wrap in a WAV container (24 kHz, mono, 16-bit)
  const pcmBytes = base64ToUint8Array(audioBase64);
  const wavBytes  = buildWAV(pcmBytes, 24000, 1, 16);

  return new Response(wavBytes, {
    status: 200,
    headers: { "content-type": "audio/wav" },
  });
}

/**
 * POST /transcribe
 * Body: raw WAV bytes (application/octet-stream)
 * Returns: JSON { transcript: string }
 */
async function handleTranscribe(request: Request, env: Env): Promise<Response> {
  const wavBytes    = new Uint8Array(await request.arrayBuffer());
  const sttModel    = env.GEMINI_STT_MODEL ?? "gemini-3-flash-preview";
  const accessToken = await getAccessToken(env);

  const audioBase64 = uint8ArrayToBase64(wavBytes);

  const vertexBody = {
    contents: [{
      role: "user",
      parts: [
        { text: "Transcribe the speech in this audio clip. Return only the transcript text, nothing else." },
        { inlineData: { mimeType: "audio/wav", data: audioBase64 } },
      ],
    }],
    generationConfig: { responseModalities: ["TEXT"] },
  };

  const endpoint = vertexEndpoint(env.GOOGLE_PROJECT_ID, sttModel);
  const upstream  = await vertexPost(endpoint, accessToken, vertexBody);

  if (!upstream.ok) {
    const errBody = await upstream.text();
    console.error(`[/transcribe] Vertex AI error ${upstream.status}: ${errBody}`);
    return new Response(errBody, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  }

  const responseJson = await upstream.json() as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string }> };
    }>;
  };

  const transcript = responseJson
    ?.candidates?.[0]
    ?.content
    ?.parts?.[0]
    ?.text ?? "";

  return new Response(JSON.stringify({ transcript: transcript.trim() }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// ── Audio helpers ──────────────────────────────────────────────────────────

/** Wraps raw PCM16 bytes in a minimal WAV container. */
function buildWAV(
  pcm: Uint8Array,
  sampleRate: number,
  channels: number,
  bitsPerSample: number
): Uint8Array {
  const byteRate   = sampleRate * channels * bitsPerSample / 8;
  const blockAlign = channels * bitsPerSample / 8;
  const dataSize   = pcm.byteLength;
  const buffer     = new ArrayBuffer(44 + dataSize);
  const view       = new DataView(buffer);

  const writeASCII = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeASCII(0,  "RIFF");
  view.setUint32(4,  36 + dataSize, true);
  writeASCII(8,  "WAVE");
  writeASCII(12, "fmt ");
  view.setUint32(16, 16, true);           // PCM subchunk size
  view.setUint16(20, 1, true);            // PCM format
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeASCII(36, "data");
  view.setUint32(40, dataSize, true);
  new Uint8Array(buffer, 44).set(pcm);

  return new Uint8Array(buffer);
}

// ── Crypto / encoding helpers ──────────────────────────────────────────────

function b64url(data: string): string {
  return btoa(data).replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
}

function arrayBufferToB64url(buffer: ArrayBuffer): string {
  return uint8ArrayToBase64(new Uint8Array(buffer))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");
}

function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

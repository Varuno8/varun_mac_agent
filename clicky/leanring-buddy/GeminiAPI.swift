//
//  GeminiAPI.swift
//  leanring-buddy
//
//  Vertex AI Gemini vision API — streaming and non-streaming.
//  Replaces ClaudeAPI.swift. Auth is handled by the Cloudflare Worker
//  proxy, which exchanges the GCP service-account credentials for a
//  short-lived access token and forwards the request to Vertex AI.
//
//  Request format: Vertex AI generateContent REST API.
//  Streaming:      server-sent events with "data: <json>\n\n" lines.
//

import Foundation

class GeminiAPI {
    private static let tlsWarmupLock = NSLock()
    private static var hasStartedTLSWarmup = false

    private let apiURL: URL
    var model: String
    private let session: URLSession

    init(proxyURL: String, model: String = "gemini-3.1-flash-lite-preview") {
        self.apiURL = URL(string: proxyURL)!
        self.model = model

        // Use .default so TLS session tickets are cached across requests.
        // Ephemeral sessions cause full handshakes on every call, leading to
        // transient -1200 errors when the payload (images) is large.
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 300
        config.waitsForConnectivity = true
        config.urlCache = nil
        config.httpCookieStorage = nil
        self.session = URLSession(configuration: config)

        warmUpTLSConnectionIfNeeded()
    }

    // MARK: - Helpers

    private func makeAPIRequest() -> URLRequest {
        var request = URLRequest(url: apiURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    /// Detects MIME type from the first bytes of image data.
    private func detectImageMimeType(for imageData: Data) -> String {
        if imageData.count >= 4 {
            let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47]
            if [UInt8](imageData.prefix(4)) == pngSignature {
                return "image/png"
            }
        }
        return "image/jpeg"
    }

    /// Warms up the TLS connection with a no-op HEAD request so the first
    /// real API call (carrying a large image payload) skips the cold handshake.
    private func warmUpTLSConnectionIfNeeded() {
        Self.tlsWarmupLock.lock()
        let shouldWarm = !Self.hasStartedTLSWarmup
        if shouldWarm { Self.hasStartedTLSWarmup = true }
        Self.tlsWarmupLock.unlock()

        guard shouldWarm,
              var components = URLComponents(url: apiURL, resolvingAgainstBaseURL: false) else { return }
        components.path = "/"
        components.query = nil
        components.fragment = nil
        guard let warmURL = components.url else { return }

        var warmRequest = URLRequest(url: warmURL)
        warmRequest.httpMethod = "HEAD"
        warmRequest.timeoutInterval = 10
        session.dataTask(with: warmRequest) { _, _, _ in }.resume()
    }

    /// Builds the Vertex AI `contents` array from labeled screenshot images and
    /// the user prompt, matching the multimodal request format.
    private func buildContents(
        images: [(data: Data, label: String)],
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String
    ) -> [[String: Any]] {
        var contents: [[String: Any]] = []

        // Prior conversation turns
        for (userPlaceholder, assistantResponse) in conversationHistory {
            contents.append([
                "role": "user",
                "parts": [["text": userPlaceholder]]
            ])
            contents.append([
                "role": "model",
                "parts": [["text": assistantResponse]]
            ])
        }

        // Current user turn: interleaved image + label pairs, then the prompt
        var currentParts: [[String: Any]] = []
        for image in images {
            currentParts.append([
                "inlineData": [
                    "mimeType": detectImageMimeType(for: image.data),
                    "data": image.data.base64EncodedString()
                ]
            ])
            currentParts.append(["text": image.label])
        }
        currentParts.append(["text": userPrompt])

        contents.append(["role": "user", "parts": currentParts])
        return contents
    }

    // MARK: - Streaming

    /// Sends a vision request to Gemini with streaming.
    /// Calls `onTextChunk` on the main actor each time new text arrives.
    /// Returns the full accumulated text and elapsed duration when done.
    func analyzeImageStreaming(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)] = [],
        userPrompt: String,
        onTextChunk: @MainActor @Sendable (String) -> Void
    ) async throws -> (text: String, duration: TimeInterval) {
        let startTime = Date()
        var request = makeAPIRequest()

        let body: [String: Any] = [
            "model": model,
            "system_instruction": ["parts": [["text": systemPrompt]]],
            "contents": buildContents(
                images: images,
                conversationHistory: conversationHistory,
                userPrompt: userPrompt
            ),
            "generationConfig": [
                "maxOutputTokens": 4096,
                "responseModalities": ["TEXT"]
            ]
        ]

        let bodyData = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = bodyData
        let payloadMB = Double(bodyData.count) / 1_048_576.0
        print("🌐 Gemini streaming request: \(String(format: "%.1f", payloadMB))MB, \(images.count) image(s)")

        let (byteStream, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(domain: "GeminiAPI", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            var errorLines: [String] = []
            for try await line in byteStream.lines { errorLines.append(line) }
            throw NSError(domain: "GeminiAPI", code: httpResponse.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "API Error (\(httpResponse.statusCode)): \(errorLines.joined(separator: "\n"))"])
        }

        // Vertex AI streaming returns SSE: "data: <json>\n\n"
        // Each JSON chunk has: candidates[0].content.parts[0].text
        var accumulatedText = ""

        for try await line in byteStream.lines {
            guard line.hasPrefix("data: ") else { continue }
            let jsonString = String(line.dropFirst(6))
            guard jsonString != "[DONE]",
                  let jsonData = jsonString.data(using: .utf8),
                  let payload = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                  let candidates = payload["candidates"] as? [[String: Any]],
                  let firstCandidate = candidates.first,
                  let content = firstCandidate["content"] as? [String: Any],
                  let parts = content["parts"] as? [[String: Any]],
                  let textChunk = parts.first?["text"] as? String
            else { continue }

            accumulatedText += textChunk
            let snapshot = accumulatedText
            await onTextChunk(snapshot)
        }

        return (text: accumulatedText, duration: Date().timeIntervalSince(startTime))
    }

    // MARK: - Non-streaming

    /// Non-streaming fallback — used when progressive display isn't needed.
    func analyzeImage(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)] = [],
        userPrompt: String
    ) async throws -> (text: String, duration: TimeInterval) {
        let startTime = Date()
        var request = makeAPIRequest()

        let body: [String: Any] = [
            "model": model,
            "system_instruction": ["parts": [["text": systemPrompt]]],
            "contents": buildContents(
                images: images,
                conversationHistory: conversationHistory,
                userPrompt: userPrompt
            ),
            "generationConfig": [
                "maxOutputTokens": 256,
                "responseModalities": ["TEXT"]
            ]
        ]

        let bodyData = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = bodyData

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "GeminiAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -1,
                          userInfo: [NSLocalizedDescriptionKey: "API Error: \(body)"])
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let candidates = json["candidates"] as? [[String: Any]],
              let content = candidates.first?["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]],
              let text = parts.first?["text"] as? String else {
            throw NSError(domain: "GeminiAPI", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
        }

        return (text: text, duration: Date().timeIntervalSince(startTime))
    }
}

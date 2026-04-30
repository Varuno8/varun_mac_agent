//
//  GeminiSTTProvider.swift
//  leanring-buddy
//
//  Upload-based speech-to-text provider using Gemini's multimodal
//  audio understanding. Buffers PCM16 audio while the user holds
//  push-to-talk, builds a WAV file on release, then POSTs it to the
//  Cloudflare Worker /transcribe endpoint which forwards it to Vertex AI.
//
//  Replaces AssemblyAIStreamingTranscriptionProvider.swift.
//  Same pattern as OpenAIAudioTranscriptionProvider (upload on release).
//

import AVFoundation
import Foundation

struct GeminiSTTProviderError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

final class GeminiSTTProvider: BuddyTranscriptionProvider {
    /// URL for the Cloudflare Worker /transcribe endpoint.
    private let transcribeProxyURL: String

    let displayName = "Gemini"
    let requiresSpeechRecognitionPermission = false
    var isConfigured: Bool { true }
    var unavailableExplanation: String? { nil }

    init(transcribeProxyURL: String) {
        self.transcribeProxyURL = transcribeProxyURL
    }

    func startStreamingSession(
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) async throws -> any BuddyStreamingTranscriptionSession {
        return GeminiSTTSession(
            transcribeProxyURL: transcribeProxyURL,
            keyterms: keyterms,
            onTranscriptUpdate: onTranscriptUpdate,
            onFinalTranscriptReady: onFinalTranscriptReady,
            onError: onError
        )
    }
}

// MARK: - Session

private final class GeminiSTTSession: BuddyStreamingTranscriptionSession {
    // Gemini is upload-based so there's no live transcript stream.
    // Give a generous fallback window so BuddyDictationManager doesn't
    // time-out before we finish uploading the WAV.
    let finalTranscriptFallbackDelaySeconds: TimeInterval = 12.0

    private static let targetSampleRate = 16_000.0

    private let transcribeProxyURL: String
    // Keyterms are sent as hints to Gemini so it transcribes app names,
    // project names, and tech terms correctly instead of mishearing them.
    private let keyterms: [String]
    private let onTranscriptUpdate: (String) -> Void
    private let onFinalTranscriptReady: (String) -> Void
    private let onError: (Error) -> Void

    private let audioPCM16Converter = BuddyPCM16AudioConverter(targetSampleRate: targetSampleRate)
    private var collectedPCM16AudioData = Data()
    private let audioCollectionQueue = DispatchQueue(label: "com.micky.gemini.stt.audio")

    private var isCancelled = false
    private var hasFinalTranscriptBeenDelivered = false

    init(
        transcribeProxyURL: String,
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        self.transcribeProxyURL = transcribeProxyURL
        self.keyterms = keyterms
        self.onTranscriptUpdate = onTranscriptUpdate
        self.onFinalTranscriptReady = onFinalTranscriptReady
        self.onError = onError
    }

    // Buffer each audio chunk as it arrives from the mic
    func appendAudioBuffer(_ audioBuffer: AVAudioPCMBuffer) {
        guard let pcm16Data = audioPCM16Converter.convertToPCM16Data(from: audioBuffer),
              !pcm16Data.isEmpty else { return }

        audioCollectionQueue.async { [weak self] in
            self?.collectedPCM16AudioData.append(pcm16Data)
        }
    }

    // Called when the user releases push-to-talk — upload and transcribe
    func requestFinalTranscript() {
        audioCollectionQueue.async { [weak self] in
            guard let self, !self.isCancelled else { return }

            let pcm16Data = self.collectedPCM16AudioData

            guard !pcm16Data.isEmpty else {
                DispatchQueue.main.async {
                    self.onFinalTranscriptReady("")
                }
                return
            }

            let wavData = BuddyWAVFileBuilder.buildWAVData(
                fromPCM16MonoAudio: pcm16Data,
                sampleRate: Int(Self.targetSampleRate)
            )

            Task { [weak self] in
                guard let self, !self.isCancelled else { return }
                await self.uploadAndTranscribe(wavData: wavData)
            }
        }
    }

    func cancel() {
        audioCollectionQueue.async { [weak self] in
            self?.isCancelled = true
        }
    }

    // MARK: - Upload

    private func uploadAndTranscribe(wavData: Data) async {
        guard let url = URL(string: transcribeProxyURL) else {
            deliverError(GeminiSTTProviderError(message: "Invalid transcription proxy URL: \(transcribeProxyURL)"))
            return
        }

        // Build a multipart-style JSON body: base64 WAV + keyterms hint.
        // The proxy uses keyterms to prepend a hint to the transcription prompt
        // so Gemini recognises app names, project names, and tech terms correctly.
        let wavBase64 = wavData.base64EncodedString()
        let bodyDict: [String: Any] = [
            "audio": wavBase64,
            "keyterms": keyterms
        ]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: bodyDict) else {
            deliverError(GeminiSTTProviderError(message: "Failed to encode transcription request"))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData
        request.timeoutInterval = 30

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard !isCancelled else { return }

            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                let body = String(data: data, encoding: .utf8) ?? "unknown"
                deliverError(GeminiSTTProviderError(
                    message: "Gemini transcription failed (HTTP \(statusCode)): \(body)"
                ))
                return
            }

            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let transcript = json["transcript"] as? String else {
                deliverError(GeminiSTTProviderError(message: "Invalid transcription response from proxy"))
                return
            }

            deliverFinalTranscript(transcript.trimmingCharacters(in: .whitespacesAndNewlines))

        } catch {
            guard !isCancelled else { return }
            deliverError(error)
        }
    }

    private func deliverFinalTranscript(_ transcript: String) {
        guard !hasFinalTranscriptBeenDelivered else { return }
        hasFinalTranscriptBeenDelivered = true
        DispatchQueue.main.async { [weak self] in
            self?.onFinalTranscriptReady(transcript)
        }
    }

    private func deliverError(_ error: Error) {
        print("[GeminiSTT] ❌ Transcription error: \(error.localizedDescription)")
        DispatchQueue.main.async { [weak self] in
            self?.onError(error)
        }
    }
}

//
//  MickyMemoryStore.swift
//  leanring-buddy
//
//  Persistent cross-session memory for Micky. Stores three categories:
//    • personal   — user preferences and facts that don't change often
//    • procedural — what action sequences worked or failed for a given app/task
//    • tool       — which action tags succeed for which app states
//
//  Memories are retrieved before each task via keyword matching and injected
//  into the system prompt as "RELEVANT PAST EXPERIENCE". After each completed
//  task ([TASK_DONE]), the task summary is stored as procedural memory so
//  future similar tasks benefit from what was learned.
//
//  Storage: ~/.micky_memory.json (plain JSON, max 300 entries, FIFO eviction).
//

import Foundation

// MARK: - Memory Entry

struct MickyMemory: Codable {
    enum MemoryCategory: String, Codable {
        /// Facts about the user — preferences, name, recurring workflows.
        case personal
        /// What action sequences worked or failed for a given app or task type.
        case procedural
        /// Which Micky action tags (APPLESCRIPT, AXCLICK, etc.) succeed for specific app states.
        case tool
    }

    let id: UUID
    let category: MemoryCategory
    /// Short words used for keyword-based retrieval (lowercased, min 3 chars).
    let keywords: [String]
    /// The memory content, written as a single natural-language sentence.
    let content: String
    let createdAt: Date
    /// Incremented each time this memory is surfaced in retrieval.
    var accessCount: Int
}

// MARK: - Memory Store

@MainActor
final class MickyMemoryStore {

    static let shared = MickyMemoryStore()

    private var memories: [MickyMemory] = []
    private let storageFileURL: URL

    /// Max memories kept on disk. Oldest are evicted when this is exceeded.
    private static let maximumStoredMemories = 300

    private init() {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        storageFileURL = URL(fileURLWithPath: "\(realUserHome)/.micky_memory.json")
        loadFromDisk()
    }

    // MARK: - Store

    func store(category: MickyMemory.MemoryCategory, content: String, keywords: [String]) {
        let newMemory = MickyMemory(
            id: UUID(),
            category: category,
            keywords: keywords.map { $0.lowercased() }.filter { $0.count >= 3 },
            content: content,
            createdAt: Date(),
            accessCount: 0
        )
        memories.append(newMemory)
        if memories.count > Self.maximumStoredMemories {
            memories.removeFirst(memories.count - Self.maximumStoredMemories)
        }
        saveToDisk()
        print("🧠 Memory stored [\(category.rawValue)]: \(content.prefix(80))")
    }

    // MARK: - Retrieve

    /// Keyword-scored retrieval. Returns up to `maximumResults` memories whose
    /// keyword sets overlap with words from the query. Ties broken by access count.
    func retrieveRelevant(for queryText: String, maximumResults: Int = 5) -> [MickyMemory] {
        let queryWords = Set(
            queryText.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count >= 3 }
        )
        guard !queryWords.isEmpty else { return [] }

        var scored: [(memory: MickyMemory, score: Int)] = []
        for memory in memories {
            let matchCount = memory.keywords.filter { keyword in
                queryWords.contains { queryWord in
                    keyword.contains(queryWord) || queryWord.contains(keyword)
                }
            }.count
            if matchCount > 0 {
                scored.append((memory, matchCount * 10 + memory.accessCount))
            }
        }

        return scored
            .sorted { $0.score > $1.score }
            .prefix(maximumResults)
            .map { $0.memory }
    }

    // MARK: - System Prompt Integration

    /// Returns a formatted block of relevant memories to append to the system
    /// prompt before the first iteration of a task. Returns an empty string
    /// when no memories match the query so it's safe to always call.
    func relevantMemoriesSystemBlock(for queryText: String) -> String {
        let relevant = retrieveRelevant(for: queryText)
        guard !relevant.isEmpty else { return "" }

        let lines = relevant
            .map { "  [\($0.category.rawValue)] \($0.content)" }
            .joined(separator: "\n")

        return """


        ── RELEVANT PAST EXPERIENCE ──
        things micky has learned from previous tasks with this user. treat these as priors — they reflect what worked or failed before. use this to pick the right approach faster:

        \(lines)
        """
    }

    // MARK: - Disk Persistence

    private func loadFromDisk() {
        guard let data = try? Data(contentsOf: storageFileURL),
              let loadedMemories = try? JSONDecoder().decode([MickyMemory].self, from: data) else {
            return
        }
        memories = loadedMemories
        print("🧠 Loaded \(memories.count) memories from \(storageFileURL.lastPathComponent)")
    }

    private func saveToDisk() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let encodedData = try? encoder.encode(memories) else { return }
        try? encodedData.write(to: storageFileURL, options: .atomic)
    }
}

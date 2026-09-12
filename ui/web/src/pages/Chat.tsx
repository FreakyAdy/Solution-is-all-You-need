import React, { useState } from "react";

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([
    { role: "assistant", content: "Hello! I am running on PHANTOM CORE with 7 hardware-transcendent innovations. How can I assist you today?" },
  ]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleSend = () => {
    if (!input.trim() || isGenerating) return;
    const userMsg = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsGenerating(true);

    // Simulate streaming token response
    let tokens = [
      "Here ", "is ", "a ", "high-speed ", "response ", "streamed ",
      "via ", "PHANTOM ", "CORE. ", "Wraith ", "layers ", "accurately ",
      "prefetched ", "MLP ", "weights ", "from ", "NVMe ", "storage ",
      "with ", "zero ", "GPU ", "stalls."
    ];
    let reply = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    let i = 0;
    const interval = setInterval(() => {
      if (i >= tokens.length) {
        clearInterval(interval);
        setIsGenerating(false);
        return;
      }
      reply += tokens[i];
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: reply };
        return next;
      });
      i++;
    }, 50);
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "3fr 1fr", gap: "24px", height: "calc(100vh - 160px)" }}>
      {/* Chat Area */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", padding: "24px" }}>
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "16px", paddingRight: "10px" }}>
          {messages.map((m, idx) => (
            <div
              key={idx}
              style={{
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                maxWidth: "75%",
                padding: "14px 18px",
                borderRadius: "12px",
                background: m.role === "user" ? "rgba(245, 158, 11, 0.15)" : "rgba(255, 255, 255, 0.05)",
                border: `1px solid ${m.role === "user" ? "var(--accent-amber)" : "var(--panel-border)"}`,
                lineHeight: "1.5",
              }}
            >
              <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "4px", textTransform: "uppercase" }}>
                {m.role}
              </div>
              <div>{m.content}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            style={{
              flex: 1,
              padding: "14px 18px",
              background: "rgba(0,0,0,0.4)",
              border: "1px solid var(--panel-border)",
              borderRadius: "10px",
              color: "#fff",
              fontFamily: "var(--font-sans)",
              fontSize: "14px",
              outline: "none",
            }}
          />
          <button
            onClick={handleSend}
            disabled={isGenerating}
            style={{
              padding: "14px 28px",
              background: "var(--accent-amber)",
              color: "#000",
              fontWeight: 700,
              borderRadius: "10px",
              opacity: isGenerating ? 0.5 : 1,
            }}
          >
            Send
          </button>
        </div>
      </div>

      {/* Stats Sidebar */}
      <div className="glass-panel" style={{ padding: "24px" }}>
        <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "16px" }}>SESSION TELEMETRY</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px", fontFamily: "var(--font-mono)", fontSize: "13px" }}>
          <div>
            <div style={{ color: "var(--text-muted)" }}>ACTIVE MODEL:</div>
            <div style={{ fontWeight: 700, color: "var(--accent-amber)", marginTop: "2px" }}>llama3:70b</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>GENERATION SPEED:</div>
            <div style={{ fontWeight: 700, color: "var(--accent-green)", marginTop: "2px" }}>4.2 tok/sec</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>KV CACHE COMPRESSION:</div>
            <div style={{ fontWeight: 700, color: "var(--accent-blue)", marginTop: "2px" }}>7.8× (Neural Cache)</div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>CONTEXT USAGE:</div>
            <div style={{ marginTop: "4px" }}>8,192 / 131,072</div>
            <div style={{ height: "4px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", marginTop: "4px" }}>
              <div style={{ width: "6%", height: "100%", background: "var(--accent-blue)" }}></div>
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-muted)" }}>ACTIVE SPARSITY:</div>
            <div style={{ fontWeight: 700, marginTop: "2px" }}>61.2% neurons bypassed</div>
          </div>
        </div>
      </div>
    </div>
  );
};

import { useState } from "react";

interface Props {
  onSend: (text: string) => void;
  loading: boolean;
}

export default function InputBox({ onSend, loading }: Props) {
  const [text, setText] = useState("");

  const handleSend = () => {
    const value = text.trim();
    if (!value || loading) return;
    onSend(value);
    setText("");
  };

  return (
    <div className="flex gap-3">
      <input
        value={text}
        disabled={loading}
        onChange={(e) => setText(e.target.value)}
        placeholder="Nhập câu hỏi về thủ tục hành chính..."
        aria-label="Nhập câu hỏi"
        className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-500 disabled:opacity-60 disabled:cursor-not-allowed"
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.nativeEvent.isComposing) handleSend();
        }}
      />

      <button
        onClick={handleSend}
        disabled={loading || !text.trim()}
        className={`px-6 py-2 rounded-lg text-white transition-all duration-500 transform
          ${loading || !text.trim()
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 active:scale-95 shadow-md"}`}
      >
        {loading ? "..." : "Gửi"}
      </button>
    </div>
  );
}

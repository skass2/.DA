import { useState } from "react";

interface Props {
  onSend: (text: string) => void;
}

export default function InputBox({ onSend }: Props) {
  const [text, setText] = useState("");

  const handleSend = () => {
    if (!text.trim()) return;
    onSend(text);
    setText("");
  };

  return (
    <div className="p-3 border-t flex gap-2">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="flex-1 border rounded px-2 py-1"
        placeholder="Nhập câu hỏi..."
      />

      <button
        onClick={handleSend}
        className="bg-blue-500 text-white px-4 rounded"
      >
        Gửi
      </button>
    </div>
  );
}

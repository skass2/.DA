import { useState } from "react";
import Message from "./Message";
import InputBox from "./InputBox";
import type { MessageType } from "../types/chat";

export default function ChatBox() {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (text: string) => {
    const userMsg: MessageType = { role: "user", text };
    setMessages((prev) => [...prev, userMsg]);

    setLoading(true);

    // MOCK API (sau này thay bằng backend)
    setTimeout(() => {
      const botMsg: MessageType = {
        role: "bot",
        text: "Đây là câu trả lời từ chatbot",
      };

      setMessages((prev) => [...prev, botMsg]);
      setLoading(false);
    }, 800);
  };

  return (
    <div className="w-[420px] h-[600px] bg-white rounded-xl shadow flex flex-col">
      <div className="p-4 font-semibold border-b">
        Chatbot Thủ tục hành chính
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.map((m, i) => (
          <Message key={i} msg={m} />
        ))}

        {loading && (
          <div className="text-sm text-gray-400">
            Đang trả lời...
          </div>
        )}
      </div>

      <InputBox onSend={handleSend} />
    </div>
  );
}

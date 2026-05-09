import type { ChatMessage } from "../types/chat";
import { useEffect, useState } from "react";

interface Props {
  message: ChatMessage;
}

export default function Message({ message }: Props) {
  const isUser = message.role === "user";
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4 transition-all duration-500`}>
      <div
        className={`max-w-[75%] px-4 py-2 rounded-2xl shadow-sm transform transition-all duration-500
          ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}
          ${isUser 
            ? "bg-blue-600 text-white rounded-tr-none" 
            : "bg-white dark:bg-gray-700 text-gray-800 dark:text-white border border-gray-200 dark:border-gray-600 rounded-tl-none"}`}
      >
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        <div className={`text-[10px] mt-1 ${isUser ? "text-blue-100" : "text-gray-400 dark:text-gray-500"}`}>
          {new Date(message.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
}
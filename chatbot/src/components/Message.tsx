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
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3 transition-all duration-500`}
    >
      <div
        className={`max-w-[70%] px-4 py-2 rounded-2xl shadow-md transform transition-all duration-500
          ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}
          ${isUser 
            ? "bg-gradient-to-r from-blue-500 to-blue-700 text-white" 
            : "bg-gray-200 dark:bg-gray-700 dark:text-white"}`}
      >
        {message.content}
        <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">
          {new Date(message.createdAt).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}

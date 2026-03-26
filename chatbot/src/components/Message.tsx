import type { MessageType } from "../types/chat";

interface Props {
  msg: MessageType;
}

export default function Message({ msg }: Props) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`px-3 py-2 rounded-lg max-w-[70%] ${
          isUser
            ? "bg-blue-500 text-white"
            : "bg-gray-200 text-black"
        }`}
      >
        {msg.text}
      </div>
    </div>
  );
}

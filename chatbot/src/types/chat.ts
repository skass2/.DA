export type Role = "user" | "bot";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
}

export type UserRole = "guest" | "user" | "admin";

export interface User {
  name: string;
  role: UserRole;
}

export type Role = "user" | "bot";

export interface MessageType {
  role: Role;
  text: string;
}

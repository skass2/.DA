export type Role = "user" | "bot";

export interface ProcedureRef {
  id?: string;
  name: string;
  display_name?: string;
}

export interface ChatSource {
  procedure_id?: string;
  procedure_name?: string;
  field?: string;
  section_type?: string;
  chunk_id?: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  suggestedQuestions?: string[];
  selectedProcedure?: ProcedureRef | null;
  sources?: ChatSource[];
  showMetadata?: boolean;
}

export interface ChatApiResponse {
  answer?: string | ChatApiResponse;
  suggested_questions?: string[];
  suggestedQuestions?: string[];
  selected_procedure?: ProcedureRef | null;
  selectedProcedure?: ProcedureRef | null;
  sources?: ChatSource[];
  show_metadata?: boolean;
  showMetadata?: boolean;
  procedure_candidates?: ProcedureRef[];
}

export type UserRole = "guest" | "user" | "admin";

export interface User {
  name: string;
  role: UserRole;
}

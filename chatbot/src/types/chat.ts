export interface ProcedureRef {
  id?: string;
  name?: string;
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
  role: "user" | "bot" | "assistant";
  content: string;
  createdAt?: number;
  suggestedQuestions?: string[];
  selectedProcedure?: ProcedureRef | null;
  sources?: ChatSource[];
  showMetadata?: boolean;
}

export interface ChatApiResponse {
  answer:
    | string
    | {
        answer?: string;
        suggested_questions?: string[];
        selected_procedure?: ProcedureRef | null;
        sources?: ChatSource[];
        show_metadata?: boolean;
        showMetadata?: boolean;
      };
  suggested_questions?: string[];
  selected_procedure?: ProcedureRef | null;
  sources?: ChatSource[];
  show_metadata?: boolean;
  showMetadata?: boolean;
}

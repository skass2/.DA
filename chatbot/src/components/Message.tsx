import type { ChatMessage, ChatSource } from "../types/chat";
import { useEffect, useMemo, useState } from "react";

interface Props {
  message: ChatMessage;
  showSuggestions?: boolean;
  onSuggestedQuestionClick?: (question: string) => void;
}

const SUGGESTION_LABEL_MAX_CHARS = 36;

function uniqueStrings(values?: string[]) {
  if (!values) return [];
  return Array.from(new Set(values.map((item) => item.trim()).filter(Boolean)));
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function truncateLabel(value: string, maxChars = SUGGESTION_LABEL_MAX_CHARS) {
  const text = value.replace(/\s+/g, " ").trim();
  if (text.length <= maxChars) return text;

  const cut = text.slice(0, maxChars + 1);
  const lastSpace = cut.lastIndexOf(" ");
  const safeCut = lastSpace > 18 ? cut.slice(0, lastSpace) : text.slice(0, maxChars);
  return `${safeCut.trim()}...`;
}

function removeProcedureNameFromSuggestion(question: string, procedureName?: string, displayName?: string) {
  let label = question.replace(/\s+/g, " ").trim();

  const names = [procedureName, displayName]
    .map((item) => item?.trim())
    .filter(Boolean) as string[];

  for (const name of names) {
    const escapedName = escapeRegExp(name);
    label = label.replace(new RegExp(`\\bthủ tục\\s+${escapedName}`, "gi"), "thủ tục");
    label = label.replace(new RegExp(escapedName, "gi"), "");
  }

  return label
    .replace(/\bthủ tục\s+thủ tục\b/gi, "thủ tục")
    .replace(/\bcho thủ tục\s+gồm\b/gi, "gồm")
    .replace(/\bthủ tục\s+có\b/gi, "có")
    .replace(/\bthủ tục\s+là\b/gi, "là")
    .replace(/\bthủ tục\s+ở\b/gi, "ở")
    .replace(/\s+/g, " ")
    .replace(/\s+\?/g, "?")
    .trim();
}

function buildSuggestionLabel(question: string, procedureName?: string, displayName?: string) {
  const raw = question.replace(/\s+/g, " ").trim();
  const lower = raw.toLowerCase();

  if ((lower.includes("hồ sơ") || lower.includes("giấy tờ")) && (lower.includes("gồm") || lower.includes("chuẩn bị"))) {
    return "Hồ sơ cần chuẩn bị gồm gì?";
  }

  if (lower.includes("phí") || lower.includes("lệ phí") || lower.includes("mất tiền")) {
    return "Có mất phí/lệ phí không?";
  }

  if (lower.includes("thời hạn") || lower.includes("bao lâu") || lower.includes("mấy ngày")) {
    return "Thời hạn giải quyết là bao lâu?";
  }

  if (lower.includes("nộp") || lower.includes("ở đâu") || lower.includes("cơ quan")) {
    return "Nộp hồ sơ ở đâu?";
  }

  if (lower.includes("trực tuyến") || lower.includes("online")) {
    return "Có nộp trực tuyến được không?";
  }

  if (lower.includes("kết quả") || lower.includes("nhận được")) {
    return "Kết quả nhận được là gì?";
  }

  if (lower.includes("điều kiện")) {
    return "Điều kiện cần có là gì?";
  }

  if (lower.includes("trình tự") || lower.includes("các bước")) {
    return "Trình tự thực hiện ra sao?";
  }

  if (lower.includes("đối tượng") || lower.includes("ai có thể")) {
    return "Ai có thể thực hiện?";
  }

  if (lower.includes("căn cứ pháp lý") || lower.includes("văn bản pháp luật")) {
    return "Căn cứ pháp lý là gì?";
  }

  const withoutProcedureName = removeProcedureNameFromSuggestion(raw, procedureName, displayName);
  return truncateLabel(withoutProcedureName || raw);
}

function buildSourcesTooltip(sources: ChatSource[], selectedProcedureName?: string) {
  const procedureNames = new Set<string>();

  sources.forEach((source) => {
    const name = source.procedure_name?.trim();
    if (name) procedureNames.add(name);
  });

  if (selectedProcedureName?.trim()) {
    procedureNames.add(selectedProcedureName.trim());
  }

  const names = Array.from(procedureNames);
  if (!names.length) return "Nguồn dữ liệu của thủ tục đang tư vấn";

  return [
    "Thủ tục đang tham khảo:",
    ...names.slice(0, 5).map((name, index) => `${index + 1}. ${name}`),
  ].join("\n");
}

function InfoIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M12 22a10 10 0 100-20 10 10 0 000 20z" />
    </svg>
  );
}

function PinIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 21s7-5.2 7-11a7 7 0 10-14 0c0 5.8 7 11 7 11z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10.5h.01" />
    </svg>
  );
}

export default function Message({ message, showSuggestions = false, onSuggestedQuestionClick }: Props) {
  const isUser = message.role === "user";
  const [visible, setVisible] = useState(false);
  const suggestions = uniqueStrings(message.suggestedQuestions).slice(0, 3);
  const sources = message.sources || [];
  const shouldShowMetadata =
    !isUser &&
    (message.showMetadata ?? true) &&
    (!!message.selectedProcedure?.name || sources.length > 0);

  const sourceTooltip = useMemo(
    () => buildSourcesTooltip(sources, message.selectedProcedure?.name),
    [sources, message.selectedProcedure?.name]
  );
  const procedureTooltip = message.selectedProcedure?.name
    ? `Đang tư vấn thủ tục: ${message.selectedProcedure.name}`
    : "";

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4 transition-all duration-500`}>
      <div className={`flex flex-col ${isUser ? "items-end" : "items-start"} max-w-[92%] sm:max-w-[82%] md:max-w-[75%]`}>
        <div
          className={`w-fit max-w-full px-4 py-2 rounded-2xl shadow-sm transform transition-all duration-500
            ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}
            ${isUser
              ? "bg-blue-600 text-white rounded-tr-none"
              : "bg-white dark:bg-gray-700 text-gray-800 dark:text-white border border-gray-200 dark:border-gray-600 rounded-tl-none"}`}
        >
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>

          {shouldShowMetadata && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-gray-100 dark:border-gray-600 pt-2 text-[11px] text-gray-500 dark:text-gray-300">
              {message.selectedProcedure?.name && (
                <span
                  title={procedureTooltip}
                  aria-label={procedureTooltip}
                  className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-blue-50 dark:bg-blue-900/30 border border-blue-100 dark:border-blue-800/60 text-blue-700 dark:text-blue-200 cursor-help"
                >
                  <PinIcon />
                  <span className="sr-only">Thủ tục đang tư vấn</span>
                </span>
              )}

              {sources.length > 0 && (
                <span
                  title={sourceTooltip}
                  aria-label={sourceTooltip || "Nguồn dữ liệu"}
                  className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-gray-50 dark:bg-gray-800/70 border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-300 cursor-help"
                >
                  <InfoIcon />
                  <span className="sr-only">Nguồn dữ liệu</span>
                </span>
              )}
            </div>
          )}

          <div className={`text-[10px] mt-1 ${isUser ? "text-blue-100" : "text-gray-400 dark:text-gray-500"}`}>
            {new Date(message.createdAt || Date.now()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </div>
        </div>

        {!isUser && showSuggestions && suggestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2 max-w-full">
            {suggestions.map((question) => {
              const label = buildSuggestionLabel(
                question,
                message.selectedProcedure?.name,
                message.selectedProcedure?.display_name
              );

              return (
                <button
                  key={question}
                  type="button"
                  onClick={() => onSuggestedQuestionClick?.(question)}
                  className="max-w-[220px] truncate whitespace-nowrap overflow-hidden text-xs sm:text-sm text-left px-3 py-2 rounded-full bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-200 border border-blue-100 dark:border-blue-800/60 transition-colors shadow-sm"
                  title={question}
                  aria-label={question}
                >
                  {truncateLabel(label)}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

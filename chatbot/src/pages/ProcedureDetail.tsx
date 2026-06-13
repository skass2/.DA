import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import UserMenu from '../components/UserMenu';

type FormFile = {
  template_id?: string;
  order?: number;
  original_name?: string;
  suggested_filename?: string;
  stored_name?: string;
  relative_path?: string;
  download_status?: string;
  source_url?: string;
  download_url?: string;
};

type DossierItem = {
  'Nhóm hồ sơ'?: string;
  'Tên giấy tờ'?: string;
  'Biểu mẫu'?: string;
  'Số lượng'?: string;
  'Mẫu đính kèm'?: FormFile[];
};

type MethodItem = {
  'Hình thức'?: string;
  'Thời hạn'?: string;
  'Phí, lệ phí'?: string;
  'Mô tả'?: string;
  'Liên kết phí'?: string[];
};

type LegalItem = {
  'Tên văn bản'?: string;
  'Số hiệu'?: string;
  'Ngày ban hành'?: string;
  'Cơ quan ban hành'?: string;
};

type ResultItem = {
  'Tên kết quả'?: string;
  'Mã kết quả'?: string;
};

type StepItem = {
  step?: number | string;
  label?: string;
  text?: string;
  content?: string;
};

type ProcedureRecord = {
  id?: string;
  old_internal_id?: string;
  search_code?: string;
  name?: string;
  detail_url?: string;
  content?: Record<string, any>;
};

const EMPTY_VALUES = new Set([
  '',
  'không',
  'không có',
  'không có thông tin',
  'không quy định',
  'chưa quy định',
  'không yêu cầu',
  '--',
]);

function cleanText(value: unknown): string {
  if (value === undefined || value === null) return '';
  return String(value).replace(/\s+/g, ' ').trim();
}

function stripVietnamese(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .replace(/Đ/g, 'D');
}

function isEmptyValue(value: unknown): boolean {
  const text = cleanText(value).toLowerCase();
  return EMPTY_VALUES.has(text);
}

function formatStructuredText(value: unknown): string {
  let text = cleanText(value);
  if (!text || isEmptyValue(text)) return '';

  const rules: Array<[RegExp, string]> = [
    [/\s*;\s*(?=\+\))/gi, '\n'],
    [/\s*;\s*(?=-\s)/gi, '\n'],
    [/\s*;\s*(?=\*\s)/gi, '\n'],
    [/\s*;\s*(?=Bước\s*\d+)/gi, '\n'],
    [/\s*;\s*(?=[a-z]\)\s)/gi, '\n'],
    [/\s*;\s*(?=\([ivxlcdm]+\)\s)/gi, '\n'],
    [/\s+(?=\+\))/gi, '\n'],
    [/\s+(?=\-\s+[A-ZÀ-Ỵ0-9])/g, '\n'],
    [/\s+(?=Bước\s*\d+\s*[:.\-]?)/gi, '\n'],
    [/\s+(?=[a-z]\)\s+[A-ZÀ-Ỵ])/g, '\n'],
    [/\s+(?=\([ivxlcdm]+\)\s+[A-ZÀ-Ỵ])/gi, '\n'],
    [/\s+(?=\+\s*Trường hợp)/gi, '\n'],
    [/\s+(?=\-\s*Trường hợp)/gi, '\n'],
  ];

  rules.forEach(([pattern, replacement]) => {
    text = text.replace(pattern, replacement);
  });

  return text
    .split('\n')
    .map(line => cleanText(line))
    .filter(Boolean)
    .join('\n')
    .trim();
}

function normalizeKey(value: unknown): string {
  let text = cleanText(value);
  text = text.replace(/^\s*[\-+•*]+\s*/, '');
  text = text.replace(/^\s*\(?[ivxlcdm]+\)\s*/i, '');
  text = text.replace(/^\s*[a-z]\)\s*/i, '');
  text = text.replace(/^\s*\d+[\).]\s*/, '');
  text = stripVietnamese(text).toLowerCase();
  text = text.replace(/\([^)]*\)/g, ' ');
  text = text.replace(/[\-+•*;:,.]+/g, ' ');
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

function trimBulletPrefix(value: unknown): string {
  let text = cleanText(value);
  text = text.replace(/^\s*[\-+•*]+\s*/, '');
  text = text.replace(/^\s*\(?[ivxlcdm]+\)\s*/i, '');
  text = text.replace(/^\s*[a-z]\)\s*/i, '');
  text = text.replace(/^\s*\d+[\).]\s*/, '');
  return text.replace(/^[;:.\-\s]+|[;:.\-\s]+$/g, '').trim();
}

function splitDisplayLines(value: unknown): string[] {
  const text = formatStructuredText(value);
  if (!text) return [];
  return text.split('\n').map(line => cleanText(line)).filter(Boolean);
}

function getContent(procedure: ProcedureRecord | null): Record<string, any> {
  return procedure?.content && typeof procedure.content === 'object' ? procedure.content : {};
}

function getProcedureName(procedure: ProcedureRecord | null): string {
  const content = getContent(procedure);
  return cleanText(procedure?.name || content['Tên thủ tục'] || 'Không rõ tên thủ tục');
}

function getApiUrl(): string {
  return import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
}

function splitStepsFromText(value: unknown): StepItem[] {
  const text = formatStructuredText(value);
  if (!text || isEmptyValue(text)) return [];

  const stepRegex = /(?:^|\n)\s*(?:[-+*]\s*)?(?:[a-z]\)\s*)?Bước\s*(\d+)\s*[:.\-]?/gi;
  const stepMatches = Array.from(text.matchAll(stepRegex));

  if (stepMatches.length > 0) {
    return stepMatches.map((match, index) => {
      const start = match.index ?? 0;
      const end = stepMatches[index + 1]?.index ?? text.length;
      let raw = text.slice(start, end).replace(/^[\s\n;+\-*]+/, '').trim();
      raw = raw.replace(/^.*?(Bước\s*\d+\s*[:.\-]?)/i, '$1').trim();
      const step = Number(match[1]) || index + 1;

      return {
        step,
        label: `Bước ${step}`,
        text: raw,
      };
    }).filter(item => cleanText(item.text));
  }

  const letterRegex = /(?:^|\n)\s*([a-z])\)\s+/gi;
  const letterMatches = Array.from(text.matchAll(letterRegex));

  if (letterMatches.length >= 2) {
    return letterMatches.map((match, index) => {
      const start = match.index ?? 0;
      const end = letterMatches[index + 1]?.index ?? text.length;
      const raw = text.slice(start, end).trim();
      const label = `Mục ${match[1].toLowerCase()})`;

      return {
        step: index + 1,
        label,
        text: raw,
      };
    }).filter(item => cleanText(item.text));
  }

  const lines = splitDisplayLines(text);
  const bulletLike = lines.filter(line => /^(\+\)|\+|-|\*|•|\([ivxlcdm]+\)|[ivxlcdm]+\))\s*/i.test(line));

  if (bulletLike.length >= 2) {
    return lines.map((line, index) => ({
      step: index + 1,
      label: `Ý ${index + 1}`,
      text: line,
    }));
  }

  const semicolonParts = cleanText(value)
    .split(/\s*;\s*/g)
    .map(part => cleanText(part))
    .filter(Boolean);

  if (semicolonParts.length >= 2) {
    return semicolonParts.map((part, index) => ({
      step: index + 1,
      label: `Ý ${index + 1}`,
      text: part,
    }));
  }

  return [{ step: 1, label: 'Trình tự thực hiện', text }];
}

function getProcedureSteps(procedure: ProcedureRecord | null): StepItem[] {
  const content = getContent(procedure);
  const prepared = content['Trình tự thực hiện_steps'];

  if (Array.isArray(prepared) && prepared.length > 0) {
    const steps = prepared
      .map((item: any, index: number) => {
        if (typeof item === 'string') {
          return {
            step: index + 1,
            label: `Bước ${index + 1}`,
            text: formatStructuredText(item),
          };
        }

        const text = formatStructuredText(item?.text || item?.content || '');
        const step = item?.step || index + 1;
        const label = cleanText(item?.label || `Bước ${step}`);

        return { step, label, text };
      })
      .filter((item: StepItem) => cleanText(item.text));

    if (steps.length > 0) return steps;
  }

  return splitStepsFromText(content['Trình tự thực hiện']);
}

function normalizeDossier(items: unknown): DossierItem[] {
  if (!Array.isArray(items)) return [];

  const seen = new Set<string>();
  const result: DossierItem[] = [];

  items.forEach((raw: any) => {
    if (!raw || typeof raw !== 'object') return;

    const tenGiayTo = trimBulletPrefix(raw['Tên giấy tờ']);
    if (!tenGiayTo || isEmptyValue(tenGiayTo)) return;

    const soLuong = cleanText(raw['Số lượng']);
    const bieuMau = cleanText(raw['Biểu mẫu']);
    const nhomHoSo = trimBulletPrefix(raw['Nhóm hồ sơ']);
    const key = normalizeKey(`${tenGiayTo}|${soLuong}|${bieuMau}`);

    if (!key || seen.has(key)) return;
    seen.add(key);

    const forms = Array.isArray(raw['Mẫu đính kèm'])
      ? raw['Mẫu đính kèm'].filter((form: FormFile) =>
          cleanText(form?.original_name || form?.suggested_filename || form?.stored_name || form?.relative_path || form?.download_url)
        )
      : [];

    result.push({
      'Nhóm hồ sơ': nhomHoSo,
      'Tên giấy tờ': tenGiayTo,
      'Biểu mẫu': bieuMau,
      'Số lượng': soLuong,
      'Mẫu đính kèm': forms,
    });
  });

  return result;
}

function normalizeLegal(items: unknown): LegalItem[] {
  if (!Array.isArray(items)) return [];

  const seen = new Set<string>();
  const result: LegalItem[] = [];

  items.forEach((raw: any) => {
    if (!raw || typeof raw !== 'object') return;

    const ten = cleanText(raw['Tên văn bản']);
    const soHieu = cleanText(raw['Số hiệu']);
    const ngay = cleanText(raw['Ngày ban hành']);
    const coQuan = cleanText(raw['Cơ quan ban hành']);
    const key = normalizeKey(`${ten}|${soHieu}|${ngay}|${coQuan}`);

    if (!key || seen.has(key)) return;
    seen.add(key);

    if (ten || soHieu) {
      result.push({
        'Tên văn bản': ten,
        'Số hiệu': soHieu,
        'Ngày ban hành': ngay,
        'Cơ quan ban hành': coQuan,
      });
    }
  });

  return result;
}

function normalizeResults(items: unknown, fallback: unknown): ResultItem[] {
  if (Array.isArray(items) && items.length > 0) {
    const seen = new Set<string>();
    return items
      .map((raw: any) => ({
        'Tên kết quả': cleanText(raw?.['Tên kết quả']),
        'Mã kết quả': cleanText(raw?.['Mã kết quả']),
      }))
      .filter(item => {
        const key = normalizeKey(`${item['Tên kết quả']}|${item['Mã kết quả']}`);
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return Boolean(item['Tên kết quả']);
      });
  }

  const text = cleanText(fallback);
  if (!text || isEmptyValue(text)) return [];
  return [{ 'Tên kết quả': text, 'Mã kết quả': '' }];
}

function InfoItem({ label, value }: { label: string; value: unknown }) {
  const text = cleanText(value);
  if (!text || isEmptyValue(text)) return null;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/40 p-4">
      <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">{label}</p>
      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 leading-relaxed">{text}</p>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="inline-flex px-4 py-2 rounded-lg bg-blue-50 dark:bg-blue-900/30 border border-blue-100 dark:border-blue-800 text-blue-700 dark:text-blue-300 text-lg font-bold">
      {children}
    </h2>
  );
}

function StructuredText({ value }: { value: unknown }) {
  const lines = splitDisplayLines(value);

  if (lines.length === 0) {
    return <p className="text-gray-500 dark:text-gray-400 italic">Chưa có thông tin.</p>;
  }

  return (
    <div className="space-y-2 text-gray-800 dark:text-gray-200 leading-7">
      {lines.map((line, index) => {
        const isBullet = /^(\+\)|\+|-|\*|•|\([ivxlcdm]+\)|[ivxlcdm]+\))\s*/i.test(line);
        const cleanedLine = line.replace(/^(\+\)|\+|-|\*|•)\s*/i, '').trim();

        return (
          <div key={`${line}-${index}`} className={isBullet ? 'flex gap-2' : ''}>
            {isBullet && <span className="mt-0.5 text-blue-600 dark:text-blue-300 font-bold">•</span>}
            <p className="whitespace-pre-wrap">{isBullet ? cleanedLine : line}</p>
          </div>
        );
      })}
    </div>
  );
}

function FormLink({ form }: { form: FormFile }) {
  const fileName = cleanText(form.original_name || form.suggested_filename || form.stored_name || 'File mẫu');
  const storedName = cleanText(form.stored_name);
  const directUrl = cleanText(form.download_url);
  const href = directUrl || (storedName ? `${getApiUrl()}/user/procedure-forms/${encodeURIComponent(storedName)}` : '');

  if (!href) {
    return (
      <span className="inline-flex items-center rounded-md bg-gray-100 dark:bg-gray-700 px-3 py-1 text-xs text-gray-600 dark:text-gray-300">
        {fileName}
      </span>
    );
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center rounded-md bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 px-3 py-1 text-xs font-semibold text-blue-700 dark:text-blue-300 border border-blue-100 dark:border-blue-800 transition-colors"
    >
      Tải file mẫu: {fileName}
    </a>
  );
}

const ProcedureDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  const [procedure, setProcedure] = useState<ProcedureRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [showTopBtn, setShowTopBtn] = useState(false);
  const [darkMode, setDarkMode] = useState(() => document.documentElement.classList.contains('dark'));

  useEffect(() => {
    const fetchProcedure = async () => {
      setLoading(true);
      try {
        const apiUrl = getApiUrl();
        const response = await fetch(`${apiUrl}/user/procedures/${encodeURIComponent(id || '')}`, {
          headers: { 'ngrok-skip-browser-warning': 'true' },
        });
        const data = await response.json();
        setProcedure(data.procedure || null);
      } catch (error) {
        console.error('Lỗi tải chi tiết thủ tục:', error);
        setProcedure(null);
      } finally {
        setLoading(false);
      }
    };

    fetchProcedure();
  }, [id]);

  const content = getContent(procedure);
  const procedureName = getProcedureName(procedure);
  const steps = useMemo(() => getProcedureSteps(procedure), [procedure]);
  const dossiers = useMemo(() => normalizeDossier(content['Thành phần hồ sơ']), [content]);
  const methods: MethodItem[] = Array.isArray(content['Cách thức thực hiện']) ? content['Cách thức thực hiện'] : [];
  const legalDocs = useMemo(() => normalizeLegal(content['Căn cứ pháp lý']), [content]);
  const results = useMemo(() => normalizeResults(content['Kết quả xử lý'], content['Kết quả thực hiện']), [content]);

  const handleAskBot = () => {
    navigate('/chat', {
      state: {
        anchorMessage: `Tôi muốn hỏi chi tiết thêm về thủ tục: ${procedureName}`,
      },
    });
  };

  const toggleDarkMode = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setShowTopBtn(e.currentTarget.scrollTop > 300);
  };

  const scrollToTop = () => {
    const container = document.getElementById('detail-scroll-container');
    if (container) container.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center text-gray-600 dark:text-gray-300">
        Đang tải chi tiết thủ tục...
      </div>
    );
  }

  if (!procedure) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col items-center justify-center gap-4 text-gray-700 dark:text-gray-200">
        <p className="text-lg font-semibold">Không tìm thấy thủ tục.</p>
        <Link to="/" className="text-blue-600 dark:text-blue-300 font-semibold hover:underline">
          Quay lại Cổng tra cứu
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      <div
        id="detail-scroll-container"
        onScroll={handleScroll}
        className="h-screen overflow-y-auto"
        style={{ scrollbarWidth: 'thin' }}
      >
        <div className="max-w-5xl mx-auto p-4 md:p-8">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6 pb-5 border-b border-gray-100 dark:border-gray-700">
              <Link to="/" className="text-blue-600 dark:text-blue-300 font-semibold hover:underline">
                Quay lại Cổng tra cứu
              </Link>

              <div className="flex items-center gap-3">
                <button
                  onClick={toggleDarkMode}
                  className="w-10 h-10 rounded-full border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  title={darkMode ? 'Bật chế độ sáng' : 'Bật chế độ tối'}
                >
                  {darkMode ? "☀️" : "🌙"}
                </button>
                <UserMenu />
              </div>
            </div>

            <div className="mb-6">
              <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white leading-snug mb-4">
                {procedureName}
              </h1>

              <div className="flex flex-wrap gap-2">
                {cleanText(procedure.search_code || content['Mã thủ tục']) && (
                  <span className="px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-semibold border border-blue-100 dark:border-blue-800">
                    Mã: {cleanText(procedure.search_code || content['Mã thủ tục'])}
                  </span>
                )}
                {cleanText(content['Lĩnh vực']) && !isEmptyValue(content['Lĩnh vực']) && (
                  <span className="px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs font-semibold">
                    Lĩnh vực: {cleanText(content['Lĩnh vực'])}
                  </span>
                )}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              <InfoItem label="Mã thủ tục đầy đủ" value={procedure.id} />
              <InfoItem label="Số quyết định" value={content['Số quyết định']} />
              <InfoItem label="Cấp thực hiện" value={content['Cấp thực hiện']} />
              <InfoItem label="Loại thủ tục" value={content['Loại thủ tục']} />
              <InfoItem label="Đối tượng thực hiện" value={content['Đối tượng thực hiện']} />
              <InfoItem label="Cơ quan thực hiện" value={content['Cơ quan thực hiện'] || content['Cơ quan có thẩm quyền']} />
              <InfoItem label="Địa chỉ tiếp nhận hồ sơ" value={content['Địa chỉ tiếp nhận HS']} />
              <InfoItem label="Cơ quan phối hợp" value={content['Cơ quan phối hợp']} />
            </div>
          </div>

          {results.length > 0 && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Kết quả thực hiện</SectionTitle>
              <div className="mt-4 space-y-3">
                {results.map((item, index) => (
                  <div key={index} className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700/40">
                    <StructuredText value={item['Tên kết quả']} />
                    {item['Mã kết quả'] && !isEmptyValue(item['Mã kết quả']) && (
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Mã kết quả: {item['Mã kết quả']}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {steps.length > 0 && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Trình tự thực hiện</SectionTitle>
              <div className="mt-5 space-y-4">
                {steps.map((step, index) => (
                  <div key={`${step.label}-${index}`} className="rounded-xl border border-blue-100 dark:border-blue-900/60 bg-blue-50/40 dark:bg-blue-900/10 p-4">
                    <p className="text-sm font-bold text-blue-700 dark:text-blue-300 mb-2">
                      {cleanText(step.label || `Ý ${index + 1}`)}
                    </p>
                    <StructuredText value={step.text || step.content} />
                  </div>
                ))}
              </div>
            </section>
          )}

          {!isEmptyValue(content['Yêu cầu điều kiện']) && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Yêu cầu điều kiện</SectionTitle>
              <div className="mt-4">
                <StructuredText value={content['Yêu cầu điều kiện']} />
              </div>
            </section>
          )}

          {methods.length > 0 && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Cách thức thực hiện</SectionTitle>
              <div className="mt-5 grid md:grid-cols-2 gap-4">
                {methods.map((method, index) => (
                  <div key={index} className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700/40">
                    <p className="text-base font-bold text-gray-900 dark:text-white mb-3">
                      {cleanText(method['Hình thức']) || `Cách thức ${index + 1}`}
                    </p>
                    <InfoItem label="Thời hạn" value={method['Thời hạn']} />
                    <div className="mt-3">
                      <InfoItem label="Phí, lệ phí" value={method['Phí, lệ phí']} />
                    </div>
                    {!isEmptyValue(method['Mô tả']) && (
                      <div className="mt-3 text-sm">
                        <StructuredText value={method['Mô tả']} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {dossiers.length > 0 && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Thành phần hồ sơ</SectionTitle>
              <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">
                Đã lọc các dòng trùng lặp để danh sách gọn hơn.
              </p>
              <div className="mt-5 space-y-3">
                {dossiers.map((item, index) => (
                  <div key={`${normalizeKey(item['Tên giấy tờ'])}-${index}`} className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700/40">
                    <p className="font-semibold text-gray-900 dark:text-white leading-relaxed">
                      {index + 1}. {cleanText(item['Tên giấy tờ'])}
                    </p>

                    {item['Nhóm hồ sơ'] && !isEmptyValue(item['Nhóm hồ sơ']) && item['Nhóm hồ sơ'] !== '--' && (
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                        Nhóm hồ sơ: {item['Nhóm hồ sơ']}
                      </p>
                    )}

                    {item['Số lượng'] && !isEmptyValue(item['Số lượng']) && (
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">
                        Số lượng: <span className="font-semibold">{item['Số lượng']}</span>
                      </p>
                    )}

                    {item['Biểu mẫu'] && !isEmptyValue(item['Biểu mẫu']) && (
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">
                        Biểu mẫu: <span className="font-semibold">{item['Biểu mẫu']}</span>
                      </p>
                    )}

                    {Array.isArray(item['Mẫu đính kèm']) && item['Mẫu đính kèm'].length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-3">
                        {item['Mẫu đính kèm'].map((form, formIndex) => (
                          <FormLink key={`${form.template_id || form.stored_name || formIndex}`} form={form} />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {legalDocs.length > 0 && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-6">
              <SectionTitle>Căn cứ pháp lý</SectionTitle>
              <div className="mt-5 space-y-3">
                {legalDocs.map((law, index) => (
                  <div key={`${law['Số hiệu']}-${index}`} className="rounded-xl border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-700/40">
                    <p className="font-semibold text-gray-900 dark:text-white">{law['Tên văn bản'] || law['Số hiệu']}</p>
                    {law['Số hiệu'] && <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">Số hiệu: {law['Số hiệu']}</p>}
                    {law['Ngày ban hành'] && !isEmptyValue(law['Ngày ban hành']) && <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">Ngày ban hành: {law['Ngày ban hành']}</p>}
                    {law['Cơ quan ban hành'] && !isEmptyValue(law['Cơ quan ban hành']) && <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">Cơ quan ban hành: {law['Cơ quan ban hành']}</p>}
                  </div>
                ))}
              </div>
            </section>
          )}

          {(content['source_url'] || procedure.detail_url) && (
            <section className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5 md:p-8 mb-24">
              <SectionTitle>Nguồn dữ liệu</SectionTitle>
              <div className="mt-4">
                <a
                  href={cleanText(content['source_url'] || procedure.detail_url)}
                  target="_blank"
                  rel="noreferrer"
                  className="text-blue-600 dark:text-blue-300 font-semibold hover:underline break-all"
                >
                  Xem trên Cổng Dịch vụ công Quốc gia
                </a>
              </div>
            </section>
          )}
        </div>
      </div>

      <button
        onClick={handleAskBot}
        className="fixed right-5 bottom-5 z-40 rounded-full bg-blue-600 hover:bg-blue-700 text-white px-5 py-3 shadow-lg font-bold transition-colors"
      >
        Hỏi AI về thủ tục này
      </button>

      {showTopBtn && (
        <button
          onClick={scrollToTop}
          className="fixed right-5 bottom-20 z-40 rounded-full bg-gray-600 hover:bg-gray-700 text-white px-4 py-3 shadow-lg font-semibold transition-colors"
        >
          Lên đầu
        </button>
      )}
    </div>
  );
};

export default ProcedureDetail;

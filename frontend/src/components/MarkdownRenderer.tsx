import { Fragment, ReactNode, useMemo, useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
  tone?: 'default' | 'inverse';
}

type MarkdownTone = NonNullable<MarkdownRendererProps['tone']>;

type MarkdownBlock =
  | { type: 'heading'; level: number; content: string }
  | { type: 'paragraph'; content: string }
  | { type: 'unordered-list'; items: string[] }
  | { type: 'ordered-list'; items: string[] }
  | { type: 'code'; language?: string; content: string }
  | { type: 'blockquote'; content: string }
  | { type: 'table'; headers: string[]; rows: string[][] }
  | { type: 'hr' };

const BLOCK_START_PATTERNS = [
  /^```/,
  /^#{1,6}\s+/,
  /^>\s?/,
  /^(\s*)[-+*]\s+/,
  /^\d+\.\s+/,
  /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
];

const TONE_STYLES: Record<MarkdownTone, Record<string, string>> = {
  default: {
    root: 'text-foreground/95',
    heading: 'text-foreground',
    link: 'text-sky-600 dark:text-sky-400 hover:text-sky-500 dark:hover:text-sky-300',
    inlineCode: 'border border-border/70 bg-muted/80 text-foreground shadow-sm',
    codeShell: 'border border-border/80 bg-[#0b1020] text-slate-100 shadow-sm',
    codeHeader: 'border-b border-white/10 bg-black/20 text-slate-300',
    codeButton: 'text-slate-300 hover:text-white hover:bg-white/10',
    quote: 'border-l-4 border-sky-500/40 bg-sky-500/5 text-foreground/85',
    hr: 'border-border/70',
    tableWrap: 'border border-border/80 bg-background/90 shadow-sm',
    tableHead: 'bg-muted/70 text-foreground',
    tableRow: 'border-t border-border/70',
    tableCell: 'text-foreground/90'
  },
  inverse: {
    root: 'text-primary-foreground/95',
    heading: 'text-primary-foreground',
    link: 'text-primary-foreground underline decoration-primary-foreground/70 hover:decoration-primary-foreground',
    inlineCode: 'border border-white/15 bg-black/20 text-primary-foreground',
    codeShell: 'border border-white/10 bg-black/35 text-primary-foreground',
    codeHeader: 'border-b border-white/10 bg-black/25 text-primary-foreground/75',
    codeButton: 'text-primary-foreground/75 hover:text-primary-foreground hover:bg-white/10',
    quote: 'border-l-4 border-white/35 bg-white/10 text-primary-foreground/85',
    hr: 'border-white/20',
    tableWrap: 'border border-white/10 bg-black/15',
    tableHead: 'bg-white/10 text-primary-foreground',
    tableRow: 'border-t border-white/10',
    tableCell: 'text-primary-foreground/90'
  }
};

const normalizeMarkdown = (content: string) => content.replace(/\r\n?/g, '\n');

const isBlockBoundary = (line: string) => BLOCK_START_PATTERNS.some(pattern => pattern.test(line));

const isTableSeparator = (line?: string) => Boolean(line && /^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line));

const isTableStart = (line?: string, nextLine?: string) => Boolean(line && nextLine && line.includes('|') && isTableSeparator(nextLine));

const splitTableLine = (line: string) => {
  let normalized = line.trim();
  if (normalized.startsWith('|')) normalized = normalized.slice(1);
  if (normalized.endsWith('|')) normalized = normalized.slice(0, -1);
  return normalized.split('|').map(cell => cell.trim());
};

function renderInline(text: string, keyPrefix: string, tone: MarkdownTone): ReactNode[] {
  const tokens: ReactNode[] = [];
  const pattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|\*([^*\n]+)\*|_([^_\n]+)_/g;
  const styles = TONE_STYLES[tone];
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      tokens.push(text.slice(cursor, match.index));
    }

    const [matched] = match;
    if (match[1] && match[2]) {
      tokens.push(
        <a
          key={`${keyPrefix}-link-${match.index}`}
          href={match[2]}
          target="_blank"
          rel="noreferrer"
          className={`font-medium underline decoration-1 underline-offset-4 transition-colors break-all ${styles.link}`}
        >
          {renderInline(match[1], `${keyPrefix}-link-text-${match.index}`, tone)}
        </a>
      );
    } else if (match[3]) {
      tokens.push(
        <code
          key={`${keyPrefix}-code-${match.index}`}
          className={`rounded-md px-1.5 py-0.5 font-mono text-[0.92em] ${styles.inlineCode}`}
        >
          {match[3]}
        </code>
      );
    } else if (match[4] || match[5]) {
      const strongText = match[4] || match[5] || '';
      tokens.push(
        <strong key={`${keyPrefix}-strong-${match.index}`} className="font-semibold">
          {renderInline(strongText, `${keyPrefix}-strong-text-${match.index}`, tone)}
        </strong>
      );
    } else if (match[6]) {
      tokens.push(
        <del key={`${keyPrefix}-del-${match.index}`} className="opacity-80">
          {renderInline(match[6], `${keyPrefix}-del-text-${match.index}`, tone)}
        </del>
      );
    } else if (match[7] || match[8]) {
      const emText = match[7] || match[8] || '';
      tokens.push(
        <em key={`${keyPrefix}-em-${match.index}`} className="italic">
          {renderInline(emText, `${keyPrefix}-em-text-${match.index}`, tone)}
        </em>
      );
    } else {
      tokens.push(matched);
    }

    cursor = match.index + matched.length;
  }

  if (cursor < text.length) {
    tokens.push(text.slice(cursor));
  }

  return tokens;
}

function parseBlocks(content: string): MarkdownBlock[] {
  const lines = normalizeMarkdown(content).split('\n');
  const blocks: MarkdownBlock[] = [];
  let index = 0;

  while (index < lines.length) {
    const currentLine = lines[index];

    if (!currentLine.trim()) {
      index += 1;
      continue;
    }

    const codeStart = currentLine.match(/^```\s*([^`]*)\s*$/);
    if (codeStart) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length && /^```\s*$/.test(lines[index])) {
        index += 1;
      }
      blocks.push({
        type: 'code',
        language: codeStart[1]?.trim() || undefined,
        content: codeLines.join('\n')
      });
      continue;
    }

    const heading = currentLine.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      blocks.push({ type: 'heading', level: heading[1].length, content: heading[2].trim() });
      index += 1;
      continue;
    }

    if (/^ {0,3}([-*_])(?:\s*\1){2,}\s*$/.test(currentLine)) {
      blocks.push({ type: 'hr' });
      index += 1;
      continue;
    }

    if (isTableStart(currentLine, lines[index + 1])) {
      const headers = splitTableLine(currentLine);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
        rows.push(splitTableLine(lines[index]));
        index += 1;
      }
      blocks.push({ type: 'table', headers, rows });
      continue;
    }

    if (/^>\s?/.test(currentLine)) {
      const quoteLines: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^>\s?/, ''));
        index += 1;
      }
      blocks.push({ type: 'blockquote', content: quoteLines.join('\n') });
      continue;
    }

    if (/^(\s*)[-+*]\s+/.test(currentLine)) {
      const items: string[] = [];
      while (index < lines.length) {
        const listLine = lines[index];
        const itemMatch = listLine.match(/^(\s*)[-+*]\s+(.*)$/);
        if (itemMatch) {
          items.push(itemMatch[2]);
          index += 1;
          continue;
        }
        if (!listLine.trim()) {
          index += 1;
          break;
        }
        break;
      }
      blocks.push({ type: 'unordered-list', items });
      continue;
    }

    if (/^\d+\.\s+/.test(currentLine)) {
      const items: string[] = [];
      while (index < lines.length) {
        const listLine = lines[index];
        const itemMatch = listLine.match(/^\d+\.\s+(.*)$/);
        if (itemMatch) {
          items.push(itemMatch[1]);
          index += 1;
          continue;
        }
        if (!listLine.trim()) {
          index += 1;
          break;
        }
        break;
      }
      blocks.push({ type: 'ordered-list', items });
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length && lines[index].trim()) {
      if (paragraphLines.length > 0 && (isBlockBoundary(lines[index]) || isTableStart(lines[index], lines[index + 1]))) {
        break;
      }
      paragraphLines.push(lines[index]);
      index += 1;
    }
    blocks.push({ type: 'paragraph', content: paragraphLines.join('\n') });
  }

  return blocks;
}

function headingClassName(level: number) {
  if (level === 1) return 'text-[1.5rem] leading-tight font-semibold tracking-tight';
  if (level === 2) return 'text-[1.25rem] leading-tight font-semibold tracking-tight';
  if (level === 3) return 'text-[1.05rem] leading-tight font-semibold';
  return 'text-[0.95rem] leading-tight font-semibold uppercase tracking-wide';
}

function CodeBlock({ code, language, tone }: { code: string; language?: string; tone: MarkdownTone }) {
  const [copied, setCopied] = useState(false);
  const styles = TONE_STYLES[tone];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch (error) {
      console.error('Failed to copy code block', error);
    }
  };

  return (
    <div className={`overflow-hidden rounded-2xl ${styles.codeShell}`}>
      <div className={`flex items-center justify-between gap-3 px-3 py-2 text-[11px] ${styles.codeHeader}`}>
        <span className="truncate font-mono uppercase tracking-[0.18em]">{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] transition-colors ${styles.codeButton}`}
          title="复制代码"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-4 text-[13px] leading-6 font-mono whitespace-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function renderBlocks(blocks: MarkdownBlock[], keyPrefix: string, tone: MarkdownTone): ReactNode[] {
  const styles = TONE_STYLES[tone];

  return blocks.map((block, index) => {
    const key = `${keyPrefix}-${index}`;

    switch (block.type) {
      case 'heading':
        return (
          <div key={key} className={`${headingClassName(block.level)} ${styles.heading}`}>
            {renderInline(block.content, `${key}-heading`, tone)}
          </div>
        );
      case 'paragraph':
        return (
          <p key={key} className="whitespace-pre-wrap break-words text-[15px] leading-7">
            {renderInline(block.content, `${key}-paragraph`, tone)}
          </p>
        );
      case 'unordered-list':
        return (
          <ul key={key} className="list-disc space-y-2 pl-6 text-[15px] leading-7 marker:opacity-70">
            {block.items.map((item, itemIndex) => (
              <li key={`${key}-item-${itemIndex}`} className="break-words pl-1">
                {renderInline(item, `${key}-item-${itemIndex}`, tone)}
              </li>
            ))}
          </ul>
        );
      case 'ordered-list':
        return (
          <ol key={key} className="list-decimal space-y-2 pl-6 text-[15px] leading-7 marker:opacity-70">
            {block.items.map((item, itemIndex) => (
              <li key={`${key}-item-${itemIndex}`} className="break-words pl-1">
                {renderInline(item, `${key}-item-${itemIndex}`, tone)}
              </li>
            ))}
          </ol>
        );
      case 'code':
        return <CodeBlock key={key} code={block.content} language={block.language} tone={tone} />;
      case 'blockquote':
        return (
          <div key={key} className={`rounded-r-2xl px-4 py-3 ${styles.quote}`}>
            <div className="space-y-3">
              {renderBlocks(parseBlocks(block.content), `${key}-quote`, tone)}
            </div>
          </div>
        );
      case 'table':
        return (
          <div key={key} className={`overflow-x-auto rounded-2xl ${styles.tableWrap}`}>
            <table className="min-w-full border-collapse text-left text-sm">
              <thead className={styles.tableHead}>
                <tr>
                  {block.headers.map((header, headerIndex) => (
                    <th key={`${key}-header-${headerIndex}`} className="px-4 py-3 font-semibold whitespace-nowrap">
                      {renderInline(header, `${key}-header-${headerIndex}`, tone)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, rowIndex) => (
                  <tr key={`${key}-row-${rowIndex}`} className={styles.tableRow}>
                    {row.map((cell, cellIndex) => (
                      <td key={`${key}-cell-${rowIndex}-${cellIndex}`} className={`px-4 py-3 align-top whitespace-pre-wrap ${styles.tableCell}`}>
                        {renderInline(cell, `${key}-cell-${rowIndex}-${cellIndex}`, tone)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      case 'hr':
        return <hr key={key} className={styles.hr} />;
      default:
        return <Fragment key={key} />;
    }
  });
}

export function MarkdownRenderer({ content, className = '', tone = 'default' }: MarkdownRendererProps) {
  const trimmed = content.trim();
  const blocks = useMemo(() => parseBlocks(content), [content]);
  if (!trimmed) return null;

  return (
    <div className={`space-y-4 break-words text-left ${TONE_STYLES[tone].root} ${className}`.trim()}>
      {renderBlocks(blocks, 'markdown', tone)}
    </div>
  );
}

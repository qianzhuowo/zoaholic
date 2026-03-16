export type PlaygroundAttachmentKind = 'image' | 'document' | 'text';

export interface PlaygroundAttachment {
  id: string;
  name: string;
  mimeType: string;
  size: number;
  dataUrl: string;
  previewText?: string;
  kind: PlaygroundAttachmentKind;
}

const TEXT_LIKE_MIME_TYPES = new Set([
  'application/json',
  'application/xml',
  'application/yaml',
  'application/x-yaml',
  'text/csv',
  'text/markdown'
]);

const EXTENSION_MIME_FALLBACK: Record<string, string> = {
  txt: 'text/plain',
  md: 'text/markdown',
  markdown: 'text/markdown',
  json: 'application/json',
  csv: 'text/csv',
  pdf: 'application/pdf',
  yaml: 'application/yaml',
  yml: 'application/yaml',
  xml: 'application/xml'
};

export const MAX_PLAYGROUND_ATTACHMENT_COUNT = 5;
export const MAX_PLAYGROUND_ATTACHMENT_SIZE = 10 * 1024 * 1024;

export function inferAttachmentMimeType(file: File): string {
  if (file.type) {
    return file.type;
  }
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  return EXTENSION_MIME_FALLBACK[ext] || 'application/octet-stream';
}

export function isTextLikeMimeType(mimeType: string): boolean {
  return mimeType.startsWith('text/') || TEXT_LIKE_MIME_TYPES.has(mimeType);
}

export function getAttachmentKind(mimeType: string): PlaygroundAttachmentKind {
  if (mimeType.startsWith('image/')) {
    return 'image';
  }
  if (isTextLikeMimeType(mimeType)) {
    return 'text';
  }
  return 'document';
}

export function isSupportedPlaygroundAttachment(file: File): boolean {
  const mimeType = inferAttachmentMimeType(file);
  return mimeType.startsWith('image/') || mimeType === 'application/pdf' || isTextLikeMimeType(mimeType);
}

export function formatAttachmentSize(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function decodeDataUrlToBytes(dataUrl: string): Uint8Array {
  const encoded = dataUrl.split(',', 2)[1] || '';
  const binary = window.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export function decodeAttachmentText(dataUrl: string): string {
  const bytes = decodeDataUrlToBytes(dataUrl);
  const decoders = ['utf-8', 'utf-16le', 'utf-16be'];
  for (const encoding of decoders) {
    try {
      return new TextDecoder(encoding, { fatal: false }).decode(bytes);
    } catch {
      // ignore and try next encoding
    }
  }
  return new TextDecoder('utf-8').decode(bytes);
}

export function getAttachmentPreviewSummary(attachment: PlaygroundAttachment): string {
  if (attachment.kind === 'image') {
    return '点击查看图片预览';
  }
  if (attachment.kind === 'document') {
    return attachment.mimeType === 'application/pdf' ? '点击查看 PDF 预览' : attachment.mimeType;
  }
  const text = (attachment.previewText || '').trim();
  if (!text) {
    return '文本附件';
  }
  return text.replace(/\s+/g, ' ').slice(0, 80);
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error(`无法读取文件：${file.name}`));
    reader.readAsDataURL(file);
  });
}

export async function readPlaygroundAttachment(file: File): Promise<PlaygroundAttachment> {
  const mimeType = inferAttachmentMimeType(file);
  const dataUrl = await readFileAsDataUrl(file);
  const kind = getAttachmentKind(mimeType);
  const previewText = kind === 'text'
    ? (await file.text()).slice(0, 12000)
    : undefined;

  return {
    id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 8)}`,
    name: file.name,
    mimeType,
    size: file.size,
    dataUrl,
    previewText,
    kind
  };
}

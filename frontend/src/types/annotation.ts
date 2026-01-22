export interface Annotation {
    id: string;
    type: "rect" | "polygon";
    points?: number[]; // For polygon: [x1, y1, x2, y2, ...]
    x?: number;        // For rect
    y?: number;        // For rect
    width?: number;    // For rect
    height?: number;   // For rect
    label: string;
    color: string;
    visible?: boolean;
}

export interface LabelClass {
    id: string;
    name: string;
    color: string;
    shortcut: string;
}

export interface ImageItem {
    id: string;
    url: string;
    name: string;
    status: "completed" | "in_progress" | "pending";
    annotationsCount?: number;
    datasetId?: string | null;
    datasetFileId?: string | null;
}

export const ANNOTATION_TYPES = ['rect', 'polygon'] as const;

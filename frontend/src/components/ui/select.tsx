import * as React from "react"
import { cn } from "@/lib/utils"

interface SelectOption {
    value: string;
    label: string;
}

interface SelectContextValue {
    value?: string;
    onValueChange: (value: string) => void;
    disabled?: boolean;
    options: SelectOption[];
    setOptions: (options: SelectOption[]) => void;
}

const SelectContext = React.createContext<SelectContextValue | null>(null);

interface SelectProps {
    value?: string;
    onValueChange?: (value: string) => void;
    disabled?: boolean;
    children: React.ReactNode;
}

export function Select({ value, onValueChange, disabled, children }: SelectProps) {
    const [options, setOptions] = React.useState<SelectOption[]>([]);

    const contextValue = React.useMemo<SelectContextValue>(
        () => ({
            value,
            onValueChange: onValueChange || (() => {}),
            disabled,
            options,
            setOptions,
        }),
        [disabled, onValueChange, options, value]
    );

    return (
        <div className="relative">
            <SelectContext.Provider value={contextValue}>
                {children}
            </SelectContext.Provider>
        </div>
    )
}

interface SelectTriggerProps {
    children: React.ReactNode;
    className?: string;
}

export function SelectTrigger({ children, className }: SelectTriggerProps) {
    const ctx = React.useContext(SelectContext);
    const isDisabled = Boolean(ctx?.disabled);
    return (
        <div
            aria-disabled={isDisabled}
            className={cn(
                "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                isDisabled ? "cursor-not-allowed opacity-50" : "",
                className
            )}
        >
            {children}
        </div>
    )
}

interface SelectValueProps {
    placeholder?: string;
}

export function SelectValue({ placeholder }: SelectValueProps) {
    const ctx = React.useContext(SelectContext);
    const rawValue = (ctx?.value || "").trim();
    if (!rawValue) {
        return <span>{placeholder || ""}</span>;
    }
    const selectedLabel = ctx?.options.find((opt) => opt.value === rawValue)?.label?.trim() || "";
    return <span>{selectedLabel || placeholder || ""}</span>;
}

interface SelectContentProps {
    children: React.ReactNode;
}

function extractNodeText(node: React.ReactNode): string {
    if (typeof node === "string" || typeof node === "number") {
        return String(node);
    }
    if (Array.isArray(node)) {
        return node.map(extractNodeText).join(" ");
    }
    if (React.isValidElement(node)) {
        const props = node.props as { children?: React.ReactNode };
        return extractNodeText(props.children);
    }
    return "";
}

export function SelectContent({ children }: SelectContentProps) {
    const ctx = React.useContext(SelectContext);
    const isDisabled = Boolean(ctx?.disabled);
    
    // Extract options from children (SelectItems)
    const options = React.useMemo<SelectOption[]>(() => {
        const next: SelectOption[] = [];
        React.Children.forEach(children, (child) => {
            if (!React.isValidElement(child)) return;
            const props = child.props as Partial<SelectItemProps>;
            if (typeof props.value !== "string") return;
            const label = extractNodeText(props.children).replace(/\s+/g, " ").trim() || props.value;
            next.push({ value: props.value, label });
        });
        return next;
    }, [children]);

    React.useEffect(() => {
        if (!ctx) return;
        const prev = ctx.options || [];
        const same =
            prev.length === options.length &&
            prev.every((item, idx) => item.value === options[idx]?.value && item.label === options[idx]?.label);
        if (!same) {
            ctx.setOptions(options);
        }
    }, [ctx, options]);

    return (
        <select 
            className={cn(
                "absolute inset-0 w-full h-full opacity-0",
                isDisabled ? "cursor-not-allowed" : "cursor-pointer"
            )}
            value={ctx?.value ?? ""}
            onChange={(e) => ctx?.onValueChange(e.target.value)}
            disabled={isDisabled}
        >
            {options.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
        </select>
    )
}

interface SelectItemProps {
    value: string;
    children: React.ReactNode;
}

export function SelectItem({ value, children }: SelectItemProps) {
    return <option value={value}>{children}</option>
}

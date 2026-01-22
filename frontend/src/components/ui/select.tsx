import * as React from "react"
import { cn } from "@/lib/utils"

interface SelectContextValue {
    value?: string;
    onValueChange: (value: string) => void;
    disabled?: boolean;
}

const SelectContext = React.createContext<SelectContextValue | null>(null);

interface SelectProps {
    value?: string;
    onValueChange?: (value: string) => void;
    disabled?: boolean;
    children: React.ReactNode;
}

export function Select({ value, onValueChange, disabled, children }: SelectProps) {
    return (
        <div className="relative">
            <SelectContext.Provider value={{ value, onValueChange: onValueChange || (() => {}), disabled }}>
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
    return <span>{ctx?.value || placeholder}</span>
}

interface SelectContentProps {
    children: React.ReactNode;
}

export function SelectContent({ children }: SelectContentProps) {
    const ctx = React.useContext(SelectContext);
    const isDisabled = Boolean(ctx?.disabled);
    
    // Extract options from children (SelectItems)
    const options: { value: string; label: string }[] = [];
    React.Children.forEach(children, (child) => {
        if (!React.isValidElement(child)) return;
        const props = child.props as Partial<SelectItemProps>;
        if (typeof props.value !== "string") return;
        const labelNode = props.children;
        const label = typeof labelNode === "string" || typeof labelNode === "number" ? String(labelNode) : "";
        options.push({ value: props.value, label });
    });

    return (
        <select 
            className={cn(
                "absolute inset-0 w-full h-full opacity-0",
                isDisabled ? "cursor-not-allowed" : "cursor-pointer"
            )}
            value={ctx?.value}
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

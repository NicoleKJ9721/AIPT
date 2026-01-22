import * as React from "react"
import { cn } from "@/lib/utils"

interface TabsContextValue {
    value: string;
    onValueChange: (v: string) => void;
}

const TabsContext = React.createContext<TabsContextValue | null>(null);

interface TabsProps {
    defaultValue?: string;
    value?: string;
    onValueChange?: (value: string) => void;
    children: React.ReactNode;
    className?: string;
}

export function Tabs({ defaultValue, value, onValueChange, children, className }: TabsProps) {
    const [stateValue, setStateValue] = React.useState(defaultValue || "");
    const currentValue = value !== undefined ? value : stateValue;
    const changeHandler = onValueChange || setStateValue;

    return (
        <TabsContext.Provider value={{ value: currentValue, onValueChange: changeHandler }}>
            <div className={className}>{children}</div>
        </TabsContext.Provider>
    )
}

interface TabsListProps {
    className?: string;
    children: React.ReactNode;
}

export function TabsList({ className, children }: TabsListProps) {
    return (
        <div className={cn("inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground", className)}>
            {children}
        </div>
    )
}

interface TabsTriggerProps {
    value: string;
    children: React.ReactNode;
    className?: string;
}

export function TabsTrigger({ value, children, className }: TabsTriggerProps) {
    const ctx = React.useContext(TabsContext);
    const isActive = ctx?.value === value;
    return (
        <button
            className={cn(
                "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                isActive ? "bg-background text-foreground shadow-sm" : "hover:bg-background/50 hover:text-foreground",
                className
            )}
            onClick={() => ctx?.onValueChange(value)}
        >
            {children}
        </button>
    )
}

interface TabsContentProps {
    value: string;
    children: React.ReactNode;
    className?: string;
}

export function TabsContent({ value, children, className }: TabsContentProps) {
    const ctx = React.useContext(TabsContext);
    if (ctx?.value !== value) return null;
    return <div className={cn("mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", className)}>{children}</div>
}

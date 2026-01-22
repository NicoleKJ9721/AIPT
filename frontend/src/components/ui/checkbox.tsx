import * as React from "react"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

const Checkbox = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <div className="relative flex items-center">
        <input
            type="checkbox"
            className={cn(
                "peer h-4 w-4 shrink-0 rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground appearance-none cursor-pointer checked:bg-primary checked:border-primary",
                className
            )}
            ref={ref}
            {...props}
        />
        <Check className="absolute left-0 top-0 h-4 w-4 hidden peer-checked:block text-primary-foreground pointer-events-none" />
    </div>
  )
)
Checkbox.displayName = "Checkbox"

export { Checkbox }

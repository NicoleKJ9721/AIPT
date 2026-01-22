import "react";

declare module "react" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface InputHTMLAttributes<T> {
    /**
     * Non-standard attribute supported by Chromium-based browsers to enable directory picking.
     * Kept for local dataset folder import UX.
     */
    webkitdirectory?: boolean | string;
    /**
     * Non-standard attribute sometimes paired with `webkitdirectory`.
     */
    directory?: boolean | string;
  }
}

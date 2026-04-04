import Link from "next/link";

type Variant = "error" | "warning" | "info" | "loading";

interface StatusScreenProps {
  variant: Variant;
  title: string;
  message?: string;
  action?: { label: string; href: string } | { label: string; onClick: () => void };
  secondaryAction?: { label: string; href: string } | { label: string; onClick: () => void };
}

const ICON: Record<Variant, JSX.Element> = {
  error: (
    <div className="w-12 h-12 rounded-full bg-team-red/10 flex items-center justify-center">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--team-red)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
    </div>
  ),
  warning: (
    <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D97706" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
    </div>
  ),
  info: (
    <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    </div>
  ),
  loading: (
    <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin" />
  ),
};

function ActionButton({ action, primary }: { action: NonNullable<StatusScreenProps["action"]>; primary?: boolean }) {
  const className = primary
    ? "px-5 py-2.5 bg-accent text-white text-sm font-semibold rounded-xl hover:bg-accent-hover transition-colors"
    : "px-5 py-2.5 bg-background border border-border text-sm font-medium text-foreground rounded-xl hover:border-accent hover:text-accent transition-colors";

  if ("href" in action) {
    return <Link href={action.href} className={className}>{action.label}</Link>;
  }
  return <button onClick={action.onClick} className={className}>{action.label}</button>;
}

export default function StatusScreen({ variant, title, message, action, secondaryAction }: StatusScreenProps) {
  return (
    <div className="min-h-[100dvh] flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4 text-center max-w-sm px-6">
        {ICON[variant]}
        <div className="flex flex-col gap-1">
          <h2 className="text-lg font-semibold text-foreground">{title}</h2>
          {message && <p className="text-sm text-text-secondary leading-relaxed">{message}</p>}
        </div>
        {(action || secondaryAction) && (
          <div className="flex items-center gap-3 mt-1">
            {action && <ActionButton action={action} primary />}
            {secondaryAction && <ActionButton action={secondaryAction} />}
          </div>
        )}
      </div>
    </div>
  );
}

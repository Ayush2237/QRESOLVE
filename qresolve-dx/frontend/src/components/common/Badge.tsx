import type { ReactNode } from 'react';

export const Badge = ({ children, variant = 'default', className = '' }: { children: ReactNode; variant?: 'primary' | 'quantum' | 'warning' | 'default'; className?: string }) => {
  const styles = {
    primary: 'bg-primary-soft text-primary',
    quantum: 'bg-quantum-soft text-quantum-text',
    warning: 'bg-warning-soft text-warning-text',
    default: 'bg-gray-100 text-ink-muted',
  };
  
  return (
    <span className={`px-2 py-0.5 text-[11px] uppercase tracking-wider font-semibold rounded-sm ${styles[variant]} ${className}`}>
      {children}
    </span>
  );
};

import type { ReactNode } from 'react';

export const Card = ({ children, className = '', onClick }: { children: ReactNode; className?: string; onClick?: () => void }) => {
  return (
    <div 
      onClick={onClick} 
      className={`bg-surface border border-border rounded-none ${onClick ? 'cursor-pointer hover:border-primary transition-colors' : ''} ${className}`}
    >
      {children}
    </div>
  );
};

/**
 * AIDJ Logo - Using Modaura Logo PNG
 */

'use client';

import Image from 'next/image';

interface AIDJLogoProps {
  onRemixComplete?: boolean;
}

export function AIDJLogo({ onRemixComplete }: AIDJLogoProps) {
  return (
    <div style={{
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
    }}>
      {/* Logo Image */}
      <Image
        src="/logo.png"
        alt="Modaura Logo"
        width={220}
        height={75}
        style={{
          objectFit: 'contain',
          filter: onRemixComplete
            ? 'drop-shadow(0 0 10px rgba(224, 64, 251, 0.8)) drop-shadow(0 0 20px rgba(224, 64, 251, 0.5))'
            : 'drop-shadow(0 0 10px rgba(224, 64, 251, 0.4)) drop-shadow(0 0 20px rgba(224, 64, 251, 0.2))',
          transition: 'filter 0.5s ease',
        }}
        priority
      />
    </div>
  );
}

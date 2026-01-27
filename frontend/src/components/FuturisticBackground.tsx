/**
 * Futuristic Background - Pure Black Professional
 */

'use client';

export function FuturisticBackground() {
    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
                pointerEvents: 'none',
                background: '#000000', // Pure black
            }}
        />
    );
}

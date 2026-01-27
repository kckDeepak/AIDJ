/**
 * Navigation Bar - Netflix-style premium tabs
 */

'use client';

interface NavigationProps {
    activeTab: 'remix' | 'library' | 'profile';
    onTabChange: (tab: 'remix' | 'library' | 'profile') => void;
    isMobile?: boolean;
}

export function Navigation({ activeTab, onTabChange, isMobile = false }: NavigationProps) {
    const tabs = [
        { id: 'remix' as const, label: 'Studio', icon: '🎵' },
        { id: 'library' as const, label: 'Collection', icon: '📚' },
        { id: 'profile' as const, label: 'Profile', icon: '👤' },
    ];

    return (
        <div style={{
            display: 'flex',
            gap: isMobile ? '4px' : '8px',
            marginLeft: 'auto',
            alignItems: 'center',
        }}>
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    style={{
                        padding: isMobile ? '8px 12px' : '10px 24px',
                        background: activeTab === tab.id
                            ? 'var(--accent-bg)'
                            : 'transparent',
                        border: activeTab === tab.id
                            ? '1px solid var(--accent-border)'
                            : '1px solid transparent',
                        borderRadius: isMobile ? '8px' : '10px',
                        color: activeTab === tab.id ? 'var(--accent)' : 'var(--gray-300)',
                        fontSize: isMobile ? '12px' : '14px',
                        fontWeight: activeTab === tab.id ? '600' : '500',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        minHeight: '40px',
                    }}
                    onMouseEnter={(e) => {
                        if (activeTab !== tab.id) {
                            e.currentTarget.style.background = 'var(--bg-elevated)';
                            e.currentTarget.style.color = 'var(--gray-100)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (activeTab !== tab.id) {
                            e.currentTarget.style.background = 'transparent';
                            e.currentTarget.style.color = 'var(--gray-300)';
                        }
                    }}
                >
                    {isMobile && <span>{tab.icon}</span>}
                    {!isMobile && tab.label}
                    {isMobile && activeTab === tab.id && <span>{tab.label}</span>}
                </button>
            ))}
        </div>
    );
}


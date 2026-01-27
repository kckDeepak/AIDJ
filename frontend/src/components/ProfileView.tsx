/**
 * Profile View - Transparent boxes with logo color hover + Settings dropdown
 */

'use client';

import { useState } from 'react';
import { User, Music, Disc3, LogOut, Settings, ChevronDown, UserCog, Bell, Shield, HelpCircle } from 'lucide-react';

interface ProfileViewProps {
    totalSongs: number;
    totalRemixes: number;
}

export function ProfileView({ totalSongs, totalRemixes }: ProfileViewProps) {
    const [hoveredBox, setHoveredBox] = useState<string | null>(null);
    const [showSettings, setShowSettings] = useState(false);

    const stats = [
        { id: 'songs', label: 'Songs', value: totalSongs, icon: Music },
        { id: 'remixes', label: 'Remixes', value: totalRemixes, icon: Disc3 },
    ];

    const settingsOptions = [
        { id: 'edit-profile', label: 'Edit Profile', icon: UserCog },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'privacy', label: 'Privacy & Security', icon: Shield },
        { id: 'help', label: 'Help & Support', icon: HelpCircle },
        { id: 'logout', label: 'Logout', icon: LogOut, isDanger: true },
    ];

    return (
        <div style={{
            width: '100%',
            maxWidth: '500px',
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
            minHeight: 'min-content',
        }}>
            {/* Profile Header */}
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '16px',
            }}>
                <div style={{
                    width: '100px',
                    height: '100px',
                    borderRadius: '50%',
                    background: 'transparent',
                    border: '2px solid var(--gray-500)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.3s ease',
                }}>
                    <User size={40} color="var(--gray-300)" />
                </div>
                <h2 style={{
                    margin: 0,
                    fontSize: '24px',
                    fontWeight: '600',
                    color: 'var(--accent)'
                }}>
                    Profile
                </h2>
            </div>

            {/* Stats Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '16px',
            }}>
                {stats.map((stat) => {
                    const Icon = stat.icon;
                    const isHovered = hoveredBox === stat.id;
                    return (
                        <div
                            key={stat.id}
                            onMouseEnter={() => setHoveredBox(stat.id)}
                            onMouseLeave={() => setHoveredBox(null)}
                            style={{
                                padding: '20px',
                                background: 'transparent',
                                border: `1px solid ${isHovered ? 'var(--accent)' : 'var(--gray-500)'}`,
                                borderRadius: '12px',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '8px',
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                            }}
                        >
                            <Icon size={24} color={isHovered ? 'var(--accent)' : 'var(--gray-400)'} />
                            <div style={{
                                fontSize: '28px',
                                fontWeight: '700',
                                color: isHovered ? 'var(--accent)' : 'var(--gray-100)',
                                transition: 'color 0.3s ease',
                            }}>
                                {stat.value}
                            </div>
                            <div style={{
                                fontSize: '14px',
                                color: isHovered ? 'var(--accent)' : 'var(--gray-300)',
                                transition: 'color 0.3s ease',
                            }}>
                                {stat.label}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Settings Button with Dropdown */}
            <div style={{ position: 'relative' }}>
                <button
                    onClick={() => setShowSettings(!showSettings)}
                    onMouseEnter={() => setHoveredBox('settings')}
                    onMouseLeave={() => !showSettings && setHoveredBox(null)}
                    style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        padding: '14px 24px',
                        background: 'transparent',
                        border: `1px solid ${hoveredBox === 'settings' || showSettings ? 'var(--accent)' : 'var(--gray-500)'}`,
                        borderRadius: '10px',
                        color: hoveredBox === 'settings' || showSettings ? 'var(--accent)' : 'var(--gray-300)',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '500',
                        transition: 'all 0.3s ease',
                    }}
                >
                    <Settings size={18} />
                    Settings
                    <ChevronDown
                        size={16}
                        style={{
                            transform: showSettings ? 'rotate(180deg)' : 'rotate(0deg)',
                            transition: 'transform 0.3s ease',
                        }}
                    />
                </button>

            </div>

            {/* Settings Dropdown (inline, not absolute) */}
            {showSettings && (
                <div style={{
                    background: 'transparent',
                    border: '1px solid var(--gray-500)',
                    borderRadius: '10px',
                    overflow: 'hidden',
                }}>
                    {settingsOptions.map((option, idx) => {
                        const Icon = option.icon;
                        const isHovered = hoveredBox === option.id;
                        const isDanger = option.isDanger;
                        const isLast = idx === settingsOptions.length - 1;
                        return (
                            <button
                                key={option.id}
                                onMouseEnter={() => setHoveredBox(option.id)}
                                onMouseLeave={() => setHoveredBox(null)}
                                style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '12px',
                                    padding: '14px 16px',
                                    background: 'transparent',
                                    border: 'none',
                                    borderBottom: isLast ? 'none' : '1px solid var(--gray-600)',
                                    color: isDanger 
                                        ? (isHovered ? 'var(--danger)' : 'var(--gray-300)')
                                        : (isHovered ? 'var(--accent)' : 'var(--gray-300)'),
                                    cursor: 'pointer',
                                    fontSize: '14px',
                                    fontWeight: '500',
                                    transition: 'all 0.2s ease',
                                    textAlign: 'left',
                                }}
                            >
                                <Icon size={18} color={
                                    isDanger 
                                        ? (isHovered ? 'var(--danger)' : 'var(--gray-400)')
                                        : (isHovered ? 'var(--accent)' : 'var(--gray-400)')
                                } />
                                {option.label}
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

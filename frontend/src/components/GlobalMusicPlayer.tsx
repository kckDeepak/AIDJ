/**
 * Global Music Player - Pop-up player that works across all panels
 */

'use client';

import { useState, useRef, useEffect } from 'react';
import {
    Play, Pause, SkipBack, SkipForward,
    RotateCcw, RotateCw, Repeat, Repeat1, Shuffle,
    Music, X
} from 'lucide-react';

interface Song {
    id: string;
    name: string;
    bpm: number;
    key: string;
    file: File;
    isGenerated?: boolean;
    duration?: number;
}

type PlaybackMode = 'order' | 'loop' | 'shuffle';

interface GlobalMusicPlayerProps {
    songs: Song[];
    currentSong: Song | null;
    isPlaying: boolean;
    onPlayPause: () => void;
    onPrevious: () => void;
    onNext: () => void;
    onSeek: (time: number) => void;
    onModeChange: (mode: PlaybackMode) => void;
    onClose: () => void;
    currentTime: number;
    duration: number;
    playbackMode: PlaybackMode;
}

export function GlobalMusicPlayer({
    songs,
    currentSong,
    isPlaying,
    onPlayPause,
    onPrevious,
    onNext,
    onSeek,
    onModeChange,
    onClose,
    currentTime,
    duration,
    playbackMode,
}: GlobalMusicPlayerProps) {
    const progressRef = useRef<HTMLDivElement>(null);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth <= 768);
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    if (!currentSong) return null;

    function formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function handleProgressClick(e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) {
        if (!progressRef.current || duration === 0) return;
        const rect = progressRef.current.getBoundingClientRect();
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const percent = (clientX - rect.left) / rect.width;
        onSeek(Math.max(0, Math.min(1, percent)) * duration);
    }

    function cycleMode() {
        const modes: PlaybackMode[] = ['order', 'loop', 'shuffle'];
        const currentIndex = modes.indexOf(playbackMode);
        const nextIndex = (currentIndex + 1) % modes.length;
        onModeChange(modes[nextIndex]);
    }

    const getModeIcon = () => {
        switch (playbackMode) {
            case 'loop': return <Repeat1 size={isMobile ? 16 : 18} />;
            case 'shuffle': return <Shuffle size={isMobile ? 16 : 18} />;
            default: return <Repeat size={isMobile ? 16 : 18} />;
        }
    };

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    // Mobile Layout
    if (isMobile) {
        return (
            <div style={{
                position: 'fixed',
                bottom: 0,
                left: 0,
                right: 0,
                background: 'var(--bg-surface)',
                backdropFilter: 'blur(30px) saturate(180%)',
                WebkitBackdropFilter: 'blur(30px) saturate(180%)',
                borderTop: '1px solid rgba(255, 255, 255, 0.15)',
                zIndex: 1000,
                boxShadow: '0 -8px 32px rgba(0, 0, 0, 0.4)',
                paddingBottom: 'max(8px, env(safe-area-inset-bottom))',
            }}>
                {/* Progress Bar at Top */}
                <div
                    ref={progressRef}
                    onClick={handleProgressClick}
                    onTouchMove={handleProgressClick}
                    style={{
                        height: '4px',
                        background: 'rgba(255, 255, 255, 0.1)',
                        cursor: 'pointer',
                        position: 'relative',
                    }}
                >
                    <div style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        height: '100%',
                        width: `${progress}%`,
                        background: 'var(--accent)',
                        transition: 'width 0.1s linear',
                    }} />
                </div>

                {/* Main Controls */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    padding: '10px 12px',
                    gap: '10px',
                }}>
                    {/* Song Info - Compact */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        flex: 1,
                        minWidth: 0,
                    }}>
                        <div style={{
                            width: '42px',
                            height: '42px',
                            borderRadius: '8px',
                            background: currentSong.isGenerated
                                ? 'var(--accent)'
                                : 'var(--accent)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                        }}>
                            <Music size={20} color="#fff" />
                        </div>
                        <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{
                                fontSize: '13px',
                                fontWeight: '600',
                                color: 'var(--accent)',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}>
                                {currentSong.name}
                            </div>
                            <div style={{ fontSize: '11px', color: '#888' }}>
                                {formatTime(currentTime)} / {formatTime(duration)}
                            </div>
                        </div>
                    </div>

                    {/* Controls - Compact */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button
                            onClick={onPrevious}
                            style={{
                                background: 'transparent',
                                border: 'none',
                                color: '#e0e0e0',
                                cursor: 'pointer',
                                padding: '8px',
                                display: 'flex',
                            }}
                        >
                            <SkipBack size={22} />
                        </button>

                        <button
                            onClick={onPlayPause}
                            style={{
                                width: '44px',
                                height: '44px',
                                borderRadius: '50%',
                                background: 'var(--accent)',
                                border: 'none',
                                color: '#fff',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                boxShadow: '0 4px 15px var(--accent-glow)',
                            }}
                        >
                            {isPlaying ? <Pause size={22} /> : <Play size={22} style={{ marginLeft: '2px' }} />}
                        </button>

                        <button
                            onClick={onNext}
                            style={{
                                background: 'transparent',
                                border: 'none',
                                color: '#e0e0e0',
                                cursor: 'pointer',
                                padding: '8px',
                                display: 'flex',
                            }}
                        >
                            <SkipForward size={22} />
                        </button>
                    </div>

                    {/* Mode & Close */}
                    <div style={{ display: 'flex', gap: '6px' }}>
                        <button
                            onClick={cycleMode}
                            style={{
                                background: playbackMode !== 'order' ? 'var(--accent-bg)' : 'transparent',
                                border: 'none',
                                borderRadius: '6px',
                                color: playbackMode !== 'order' ? 'var(--accent)' : 'var(--gray-300)',
                                cursor: 'pointer',
                                padding: '8px',
                                display: 'flex',
                            }}
                        >
                            {getModeIcon()}
                        </button>
                        <button
                            onClick={onClose}
                            style={{
                                background: 'transparent',
                                border: 'none',
                                color: '#666',
                                cursor: 'pointer',
                                padding: '8px',
                                display: 'flex',
                            }}
                        >
                            <X size={18} />
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Desktop Layout
    return (
        <div style={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            height: '80px',
            background: 'var(--bg-surface)',
            backdropFilter: 'blur(30px) saturate(180%)',
            WebkitBackdropFilter: 'blur(30px) saturate(180%)',
            borderTop: '1px solid rgba(255, 255, 255, 0.15)',
            display: 'flex',
            alignItems: 'center',
            padding: '0 24px',
            gap: '20px',
            zIndex: 1000,
            boxShadow: '0 -8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.1)',
        }}>
            {/* Song Info */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                minWidth: '200px',
            }}>
                <div style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '10px',
                    background: currentSong.isGenerated
                        ? 'var(--accent)'
                        : 'var(--accent)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}>
                    <Music size={24} color="#fff" />
                </div>
                <div>
                    <div style={{
                        fontSize: '14px',
                        fontWeight: '600',
                        color: 'var(--accent)',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                    }}>
                        {currentSong.name}
                    </div>
                    <div style={{ fontSize: '12px', color: '#888' }}>
                        {currentSong.bpm} BPM • {currentSong.key}
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '8px',
            }}>
                {/* Buttons */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px',
                }}>
                    {/* Rewind 10s */}
                    <button
                        onClick={() => onSeek(Math.max(0, currentTime - 10))}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#999',
                            cursor: 'pointer',
                            padding: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'color 0.2s',
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
                        onMouseLeave={(e) => e.currentTarget.style.color = '#999'}
                    >
                        <RotateCcw size={18} />
                    </button>

                    {/* Previous */}
                    <button
                        onClick={onPrevious}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#e0e0e0',
                            cursor: 'pointer',
                            padding: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'color 0.2s',
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
                        onMouseLeave={(e) => e.currentTarget.style.color = '#e0e0e0'}
                    >
                        <SkipBack size={22} />
                    </button>

                    {/* Play/Pause */}
                    <button
                        onClick={onPlayPause}
                        style={{
                            width: '44px',
                            height: '44px',
                            borderRadius: '50%',
                            background: 'var(--accent)',
                            border: 'none',
                            color: '#fff',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            boxShadow: '0 4px 15px var(--accent-glow)',
                            transition: 'transform 0.2s, box-shadow 0.2s',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'scale(1.05)';
                            e.currentTarget.style.boxShadow = '0 6px 20px var(--accent-glow)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = '0 4px 15px var(--accent-glow)';
                        }}
                    >
                        {isPlaying ? <Pause size={22} /> : <Play size={22} style={{ marginLeft: '2px' }} />}
                    </button>

                    {/* Next */}
                    <button
                        onClick={onNext}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#e0e0e0',
                            cursor: 'pointer',
                            padding: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'color 0.2s',
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
                        onMouseLeave={(e) => e.currentTarget.style.color = '#e0e0e0'}
                    >
                        <SkipForward size={22} />
                    </button>

                    {/* Forward 10s */}
                    <button
                        onClick={() => onSeek(Math.min(duration, currentTime + 10))}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#999',
                            cursor: 'pointer',
                            padding: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'color 0.2s',
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
                        onMouseLeave={(e) => e.currentTarget.style.color = '#999'}
                    >
                        <RotateCw size={18} />
                    </button>
                </div>

                {/* Progress Bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    width: '100%',
                    maxWidth: '500px',
                }}>
                    <span style={{ fontSize: '11px', color: '#888', minWidth: '40px' }}>
                        {formatTime(currentTime)}
                    </span>
                    <div
                        ref={progressRef}
                        onClick={handleProgressClick}
                        style={{
                            flex: 1,
                            height: '4px',
                            background: 'rgba(255, 255, 255, 0.1)',
                            borderRadius: '2px',
                            cursor: 'pointer',
                            position: 'relative',
                        }}
                    >
                        <div style={{
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            height: '100%',
                            width: `${progress}%`,
                            background: 'var(--accent)',
                            borderRadius: '2px',
                            transition: 'width 0.1s linear',
                        }} />
                        <div style={{
                            position: 'absolute',
                            left: `${progress}%`,
                            top: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '12px',
                            height: '12px',
                            background: '#fff',
                            borderRadius: '50%',
                            boxShadow: '0 2px 6px rgba(0, 0, 0, 0.3)',
                        }} />
                    </div>
                    <span style={{ fontSize: '11px', color: '#888', minWidth: '40px', textAlign: 'right' }}>
                        {formatTime(duration)}
                    </span>
                </div>
            </div>

            {/* Right Side - Mode & Close */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
            }}>
                {/* Playback Mode */}
                <button
                    onClick={cycleMode}
                    title={`Mode: ${playbackMode}`}
                    style={{
                        background: playbackMode !== 'order' ? 'var(--accent-bg)' : 'transparent',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        color: playbackMode !== 'order' ? 'var(--accent)' : 'var(--gray-300)',
                        cursor: 'pointer',
                        padding: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        transition: 'all 0.2s',
                    }}
                >
                    {getModeIcon()}
                </button>

                {/* Close */}
                <button
                    onClick={onClose}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#666',
                        cursor: 'pointer',
                        padding: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        transition: 'color 0.2s',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.color = '#f87171'}
                    onMouseLeave={(e) => e.currentTarget.style.color = '#666'}
                >
                    <X size={20} />
                </button>
            </div>
        </div>
    );
}

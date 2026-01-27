/**
 * Library View - Playlist-Style with Search
 */

'use client';

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { MoreVertical, Download, Trash2, Edit2, Music, Search, GripVertical } from 'lucide-react';

interface Song {
    id: string;
    name: string;
    bpm: number;
    key: string;
    file: File;
    isGenerated?: boolean;
    duration?: number;
    addedAt?: Date;
}

interface LibraryViewProps {
    songs: Song[];
    onDelete: (filename: string) => void;
    onRename: (filename: string, newName: string) => void;
    onDownload: (filename: string) => void;
    onReorder?: (fromIndex: number, toIndex: number) => void;
    onPlay?: (song: Song) => void;
}

interface MenuPosition {
    top: number;
    left: number;
}

export function LibraryView({ songs, onDelete, onRename, onDownload, onReorder, onPlay }: LibraryViewProps) {
    const [hoveredMenuIndex, setHoveredMenuIndex] = useState<number | null>(null);
    const [menuPosition, setMenuPosition] = useState<MenuPosition | null>(null);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editName, setEditName] = useState('');
    const [searchQuery, setSearchQuery] = useState('');
    const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
    const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
    const [isMobile, setIsMobile] = useState(false);
    const menuButtonRefs = useRef<(HTMLButtonElement | null)[]>([]);

    // Check for mobile viewport
    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth <= 768);
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    // Filter songs based on search query
    const filteredSongs = songs.filter(song =>
        song.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    function formatDuration(seconds: number = 180): string {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function formatTimeAgo(date?: Date): string {
        if (!date) return 'Just now';
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        return 'Just now';
    }

    function handleMenuHover(index: number, buttonElement: HTMLButtonElement | null) {
        if (!buttonElement) return;

        const rect = buttonElement.getBoundingClientRect();
        const menuWidth = 140;
        const menuHeight = 150;

        let left = rect.right + 8;
        let top = rect.top;

        if (left + menuWidth > window.innerWidth) {
            left = rect.left - menuWidth - 8;
        }

        if (top + menuHeight > window.innerHeight) {
            top = window.innerHeight - menuHeight - 8;
        }

        if (top < 8) {
            top = 8;
        }

        setMenuPosition({ top, left });
        setHoveredMenuIndex(index);
    }

    function handleMenuLeave() {
        setHoveredMenuIndex(null);
        setMenuPosition(null);
    }

    function startRename(index: number, currentName: string) {
        setEditingIndex(index);
        setEditName(currentName);
        handleMenuLeave();
    }

    function saveRename(song: Song) {
        if (editName.trim() && editName !== song.name) {
            onRename(song.id, editName.trim());
        }
        setEditingIndex(null);
        setEditName('');
    }

    // Drag handlers for reordering
    function handleDragStart(e: React.DragEvent, index: number) {
        setDraggedIndex(index);
        e.dataTransfer.effectAllowed = 'move';
        // Add a slight delay to see the item lift
        e.dataTransfer.setDragImage(e.currentTarget, 0, 0);
    }

    function handleDragOver(e: React.DragEvent, index: number) {
        e.preventDefault();
        if (draggedIndex === null || draggedIndex === index) return;
        setDragOverIndex(index);
    }

    function handleDragLeave() {
        // Don't reset immediately - let it stay for smoother UX
    }

    function handleDrop(e: React.DragEvent, dropIndex: number) {
        e.preventDefault();
        if (draggedIndex === null || draggedIndex === dropIndex) return;

        if (onReorder) {
            onReorder(draggedIndex, dropIndex);
        }

        setDraggedIndex(null);
        setDragOverIndex(null);
    }

    function handleDragEnd() {
        setDraggedIndex(null);
        setDragOverIndex(null);
    }

    // Calculate transform for YouTube-style animation
    const ITEM_HEIGHT = 70; // Approximate row height

    function getDragTransform(index: number): React.CSSProperties {
        if (draggedIndex === null || dragOverIndex === null) return {};
        if (index === draggedIndex) return {}; // Dragged item handled separately

        // Moving down: items between drag start and hover should move UP
        if (draggedIndex < dragOverIndex) {
            if (index > draggedIndex && index <= dragOverIndex) {
                return {
                    transform: `translateY(-${ITEM_HEIGHT}px)`,
                    transition: 'transform 0.2s cubic-bezier(0.2, 0, 0, 1)',
                };
            }
        }
        // Moving up: items between hover and drag start should move DOWN
        else if (draggedIndex > dragOverIndex) {
            if (index >= dragOverIndex && index < draggedIndex) {
                return {
                    transform: `translateY(${ITEM_HEIGHT}px)`,
                    transition: 'transform 0.2s cubic-bezier(0.2, 0, 0, 1)',
                };
            }
        }

        return {
            transition: 'transform 0.2s cubic-bezier(0.2, 0, 0, 1)',
        };
    }



    function MenuPortal({ index, song }: { index: number; song: Song }) {
        if (hoveredMenuIndex !== index || !menuPosition) return null;

        return createPortal(
            <div
                onMouseEnter={() => setHoveredMenuIndex(index)}
                onMouseLeave={handleMenuLeave}
                style={{
                    position: 'fixed',
                    top: `${menuPosition.top}px`,
                    left: `${menuPosition.left}px`,
                    background: 'rgba(30, 30, 40, 0.98)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '8px',
                    padding: '4px',
                    minWidth: '140px',
                    zIndex: 99999,
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                }}
            >
                {/* Download - available for all songs */}
                <button
                    onClick={() => {
                        onDownload(song.id);
                        handleMenuLeave();
                    }}
                    style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: 'transparent',
                        border: 'none',
                        color: '#e0e0e0',
                        cursor: 'pointer',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        fontSize: '13px',
                        transition: 'background 0.2s ease',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'var(--accent-bg)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                    <Download size={14} />
                    Download
                </button>

                {/* Rename - available for all songs */}
                <button
                    onClick={() => startRename(index, song.name)}
                    style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: 'transparent',
                        border: 'none',
                        color: '#e0e0e0',
                        cursor: 'pointer',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        fontSize: '13px',
                        transition: 'background 0.2s ease',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'var(--accent-bg)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                    <Edit2 size={14} />
                    Rename
                </button>

                {/* Delete */}
                <button
                    onClick={() => {
                        if (confirm(`Delete "${song.name}"?`)) {
                            onDelete(song.id);
                        }
                        handleMenuLeave();
                    }}
                    style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: 'transparent',
                        border: 'none',
                        color: '#f87171',
                        cursor: 'pointer',
                        borderRadius: '6px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        fontSize: '13px',
                        transition: 'background 0.2s ease',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(248, 113, 113, 0.2)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                    <Trash2 size={14} />
                    Delete
                </button>
            </div>,
            document.body
        );
    }

    return (
        <div style={{
            width: '100%',
            height: '100%',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
        }}>
            {/* Header with Title and Search */}
            <div style={{ marginBottom: isMobile ? '12px' : '20px' }}>
                <h1 style={{
                    fontSize: isMobile ? '22px' : '28px',
                    fontWeight: '700',
                    margin: isMobile ? '0 0 12px 0' : '0 0 16px 0',
                    color: '#e0e0e0',
                }}>
                    <span style={{ color: 'var(--accent)' }}>Collection</span>
                </h1>

                {/* Search Bar */}
                <div style={{
                    position: 'relative',
                    width: '100%',
                }}>
                    <Search
                        size={18}
                        style={{
                            position: 'absolute',
                            left: '14px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            color: 'rgba(255, 255, 255, 0.4)',
                            pointerEvents: 'none',
                        }}
                    />
                    <input
                        type="text"
                        placeholder="Search"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '12px 14px 12px 42px',
                            background: 'rgba(255, 255, 255, 0.08)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(255, 255, 255, 0.15)',
                            borderRadius: '10px',
                            color: '#e0e0e0',
                            fontSize: '14px',
                            outline: 'none',
                            transition: 'all 0.2s ease',
                        }}
                        onFocus={(e) => {
                            e.currentTarget.style.borderColor = 'var(--accent-border)';
                            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)';
                        }}
                        onBlur={(e) => {
                            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                        }}
                    />
                </div>
            </div>

            {/* Playlist-Style List */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                paddingRight: isMobile ? '0' : '6px',
            }}>
                {filteredSongs.length === 0 ? (
                    <div style={{
                        textAlign: 'center',
                        padding: isMobile ? '40px 16px' : '60px 20px',
                        color: '#666',
                        fontSize: isMobile ? '13px' : '14px',
                    }}>
                        {searchQuery ? 'No songs match your search' : 'No songs in your library yet'}
                    </div>
                ) : (
                    filteredSongs.map((song, index) => (
                        <div
                            key={song.id}
                            draggable={!isMobile}
                            onDragStart={(e) => !isMobile && handleDragStart(e, index)}
                            onDragOver={(e) => !isMobile && handleDragOver(e, index)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => !isMobile && handleDrop(e, index)}
                            onDragEnd={handleDragEnd}
                            onClick={() => onPlay && onPlay(song)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: isMobile ? '10px' : '14px',
                                padding: isMobile ? '10px 12px' : '12px 14px',
                                background: dragOverIndex === index
                                    ? 'var(--accent-bg)'
                                    : draggedIndex === index
                                        ? 'var(--accent-bg)'
                                        : 'rgba(255, 255, 255, 0.02)',
                                borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                                borderLeft: dragOverIndex === index
                                    ? '3px solid var(--accent-border)'
                                    : '3px solid transparent',
                                cursor: isMobile ? 'pointer' : (draggedIndex === index ? 'grabbing' : 'grab'),
                                opacity: draggedIndex === index ? 0.8 : 1,
                                boxShadow: draggedIndex === index
                                    ? '0 8px 25px rgba(0, 0, 0, 0.4)'
                                    : 'none',
                                zIndex: draggedIndex === index ? 100 : 1,
                                ...getDragTransform(index),
                            }}
                            onMouseEnter={(e) => {
                                if (draggedIndex !== index) {
                                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (draggedIndex !== index && dragOverIndex !== index) {
                                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
                                }
                            }}
                        >
                            {/* Drag Handle - Hide on mobile */}
                            {!isMobile && (
                                <div
                                    style={{
                                        cursor: 'grab',
                                        color: '#666',
                                        display: 'flex',
                                        alignItems: 'center',
                                        transition: 'color 0.2s ease',
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.color = '#999';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.color = '#666';
                                    }}
                                >
                                    <GripVertical size={18} />
                                </div>
                            )}

                            {/* Icon */}
                            <div style={{
                                width: isMobile ? '36px' : '40px',
                                height: isMobile ? '36px' : '40px',
                                borderRadius: '8px',
                                background: song.isGenerated
                                    ? 'linear-gradient(135deg, #a855f7, #6366f1)'
                                    : 'var(--accent)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                flexShrink: 0,
                            }}>
                                <Music size={isMobile ? 18 : 20} color="#fff" />
                            </div>

                            {/* Song Info */}
                            <div style={{
                                flex: 1,
                                minWidth: 0,
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '4px',
                            }}>
                                {/* Title */}
                                {editingIndex === index ? (
                                    <input
                                        type="text"
                                        value={editName}
                                        onChange={(e) => setEditName(e.target.value)}
                                        onBlur={() => saveRename(song)}
                                        onClick={(e) => e.stopPropagation()}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') saveRename(song);
                                            if (e.key === 'Escape') {
                                                setEditingIndex(null);
                                                setEditName('');
                                            }
                                        }}
                                        autoFocus
                                        style={{
                                            padding: '4px 8px',
                                            background: 'rgba(255, 255, 255, 0.1)',
                                            border: '1px solid var(--accent-border)',
                                            borderRadius: '4px',
                                            color: '#e0e0e0',
                                            fontSize: '14px',
                                            fontWeight: '600',
                                            outline: 'none',
                                        }}
                                    />
                                ) : (
                                    <div style={{
                                        fontSize: '14px',
                                        fontWeight: '600',
                                        color: '#e0e0e0',
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                    }}>
                                        {song.name} {song.isGenerated && '✨'}
                                    </div>
                                )}

                                {/* Metadata Row */}
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '12px',
                                    fontSize: '12px',
                                    color: '#999',
                                }}>
                                    <span>{formatDuration(song.duration)}</span>
                                    <span>•</span>
                                    <span>{song.bpm} BPM</span>
                                    <span>•</span>
                                    <span>{song.key}</span>
                                    <span>•</span>
                                    <span style={{
                                        padding: '2px 6px',
                                        background: song.isGenerated
                                            ? 'rgba(168, 85, 247, 0.15)'
                                            : 'var(--accent-bg)',
                                        border: `1px solid ${song.isGenerated ? 'rgba(168, 85, 247, 0.3)' : 'var(--accent-border)'}`,
                                        borderRadius: '4px',
                                        fontSize: '10px',
                                        fontWeight: '600',
                                        color: song.isGenerated ? '#c084fc' : 'var(--accent)',
                                    }}>
                                        {song.isGenerated ? 'AI Remix' : 'Original'}
                                    </span>
                                    <span>•</span>
                                    <span>{formatTimeAgo(song.addedAt)}</span>
                                </div>
                            </div>

                            {/* Menu Button */}
                            <button
                                ref={(el) => { menuButtonRefs.current[index] = el; }}
                                onMouseEnter={(e) => handleMenuHover(index, e.currentTarget)}
                                onClick={(e) => e.stopPropagation()}
                                style={{
                                    padding: '8px',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    borderRadius: '6px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexShrink: 0,
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                <MoreVertical size={16} color="#999" />
                            </button>

                            <MenuPortal index={index} song={song} />
                        </div>
                    ))
                )}
            </div>

            {/* Results Count */}
            {searchQuery && (
                <div style={{
                    padding: '12px 0',
                    fontSize: '12px',
                    color: '#666',
                    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                    marginTop: '8px',
                }}>
                    {filteredSongs.length} {filteredSongs.length === 1 ? 'result' : 'results'}
                </div>
            )}
        </div>
    );
}

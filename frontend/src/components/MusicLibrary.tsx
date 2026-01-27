/**
 * Music Library Component - with Portal-based Menu and Hover Behavior
 */

'use client';

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { MoreVertical, Download, Trash2, Edit2, GripVertical } from 'lucide-react';

interface Song {
    id: string;
    name: string;
    bpm: number;
    key: string;
    file: File;
    isGenerated?: boolean;
}

interface MusicLibraryProps {
    songs: Song[];
    selectedSongIds: Set<string>;
    onToggleSelection: (id: string) => void;
    onFileUpload: (files: FileList | null) => void;
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

export function MusicLibrary({
    songs,
    selectedSongIds,
    onToggleSelection,
    onFileUpload,
    onDelete,
    onRename,
    onDownload,
    onReorder,
    onPlay,
}: MusicLibraryProps) {
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
    const [hoveredMenuIndex, setHoveredMenuIndex] = useState<number | null>(null);
    const [menuPosition, setMenuPosition] = useState<MenuPosition | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isAddMediaHovered, setIsAddMediaHovered] = useState(false);
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [editName, setEditName] = useState('');
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

    function handleDrop(e: React.DragEvent) {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) onFileUpload(e.dataTransfer.files);
    }

    function getItemStyle(index: number) {
        // No levitating effect - just return empty
        return {};
    }


    function handleMenuHover(index: number, buttonElement: HTMLButtonElement | null) {
        if (!buttonElement) return;

        const rect = buttonElement.getBoundingClientRect();
        const menuWidth = 140;
        const menuHeight = 150;

        // Calculate position with viewport bounds checking
        let left = rect.right + 8;
        let top = rect.top;

        // Check right edge
        if (left + menuWidth > window.innerWidth) {
            left = rect.left - menuWidth - 8;
        }

        // Check bottom edge
        if (top + menuHeight > window.innerHeight) {
            top = window.innerHeight - menuHeight - 8;
        }

        // Check top edge
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
    }

    function handleDragOver(e: React.DragEvent, index: number) {
        e.preventDefault();
        if (draggedIndex === null || draggedIndex === index) return;
        setDragOverIndex(index);
    }

    function handleDragLeave() {
        setDragOverIndex(null);
    }

    function handleDropReorder(e: React.DragEvent, dropIndex: number) {
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


    // Portal menu component
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
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
        }}>
            {/* Scrollable Song List */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                overflowX: 'hidden',
                paddingBottom: '10px',
                minHeight: 0,
            }}>
                {songs.map((song, index) => (
                    <div key={song.id}>
                        <div
                            draggable={!isMobile}
                            onDragStart={(e) => !isMobile && handleDragStart(e, index)}
                            onDragOver={(e) => !isMobile && handleDragOver(e, index)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => !isMobile && handleDropReorder(e, index)}
                            onDragEnd={handleDragEnd}
                            onMouseEnter={() => setHoveredIndex(index)}
                            onMouseLeave={() => setHoveredIndex(null)}
                            onClick={(e) => {
                                // Clicking anywhere else on the song box toggles selection for remix
                                const target = e.target as HTMLElement;
                                if (target.tagName !== 'INPUT' && target.tagName !== 'BUTTON' && editingIndex !== index) {
                                    if (!song.isGenerated) {
                                        onToggleSelection(song.id);
                                    }
                                }
                            }}
                            style={{
                                padding: '10px',
                                marginBottom: '6px',
                                backgroundColor: 'transparent',
                                border: dragOverIndex === index
                                    ? '1px solid var(--accent)'
                                    : hoveredIndex === index || selectedSongIds.has(song.id)
                                        ? '1px solid var(--accent)'
                                        : '1px solid var(--gray-500)',
                                borderRadius: '10px',
                                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                                boxShadow: 'none',
                                position: 'relative',
                                opacity: draggedIndex === index ? 0.5 : 1,
                                cursor: 'grab',
                                ...getItemStyle(index),
                            }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? '6px' : '8px' }}>
                                {/* Drag Handle - Hide on mobile */}
                                {!isMobile && <GripVertical size={16} color="#666" style={{ flexShrink: 0 }} />}

                                <input
                                    type="checkbox"
                                    checked={selectedSongIds.has(song.id)}
                                    onChange={() => onToggleSelection(song.id)}
                                    disabled={song.isGenerated}
                                    style={{ accentColor: 'var(--accent)', width: isMobile ? '18px' : '16px', height: isMobile ? '18px' : '16px' }}
                                />

                                <div style={{ flex: 1, minWidth: 0 }}>
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
                                                width: '100%',
                                                padding: '2px 6px',
                                                background: 'rgba(255, 255, 255, 0.1)',
                                                border: '1px solid var(--accent-border)',
                                                borderRadius: '4px',
                                                color: '#e0e0e0',
                                                fontSize: '13px',
                                                fontWeight: '600',
                                            }}
                                        />
                                    ) : (
                                        <>
                                            <div
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (onPlay) onPlay(song);
                                                }}
                                                style={{
                                                    fontWeight: '600',
                                                    fontSize: '13px',
                                                    whiteSpace: 'nowrap',
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis',
                                                    cursor: 'pointer',
                                                    padding: '2px 4px',
                                                    borderRadius: '4px',
                                                    transition: 'background 0.2s ease',
                                                }}
                                                onMouseEnter={(e) => e.currentTarget.style.background = 'var(--accent-bg)'}
                                                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                                                title="Click to play"
                                            >
                                                {song.name} {song.isGenerated && '✨'}
                                            </div>
                                            <div style={{ fontSize: '11px', color: '#999' }}>
                                                {song.bpm} BPM • {song.key}
                                            </div>
                                        </>
                                    )}
                                </div>

                                {/* Three-dot menu - ALWAYS VISIBLE, HOVER TO SHOW MENU */}
                                <button
                                    ref={(el) => menuButtonRefs.current[index] = el}
                                    onMouseEnter={(e) => handleMenuHover(index, e.currentTarget)}
                                    onClick={(e) => e.stopPropagation()}
                                    style={{
                                        padding: '4px',
                                        background: hoveredIndex === index
                                            ? 'rgba(255, 255, 255, 0.1)'
                                            : 'transparent',
                                        border: '1px solid rgba(255, 255, 255, 0.2)',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        transition: 'all 0.2s ease',
                                        opacity: hoveredIndex === index ? 1 : 0.5,
                                    }}
                                >
                                    <MoreVertical size={16} color="#e0e0e0" />
                                </button>

                                {/* Render menu via portal */}
                                <MenuPortal index={index} song={song} />
                            </div>
                        </div>

                        {/* Divider */}
                        {index < songs.length - 1 && (
                            <div style={{
                                height: '1px',
                                background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent)',
                                margin: '0 10px 6px 10px',
                            }} />
                        )}
                    </div>
                ))}

                {songs.length === 0 && (
                    <div style={{
                        textAlign: 'center',
                        color: '#666',
                        padding: isMobile ? '30px 16px' : '40px 20px',
                        fontSize: '13px'
                    }}>
                        No songs yet
                    </div>
                )}
            </div>

            {/* Fixed Bottom: Add Media */}
            <div
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onMouseEnter={() => setIsAddMediaHovered(true)}
                onMouseLeave={() => setIsAddMediaHovered(false)}
                style={{
                    marginTop: isMobile ? '10px' : '12px',
                    padding: isMobile ? '14px' : '16px',
                    textAlign: 'center',
                    background: isAddMediaHovered || isDragging
                        ? 'var(--accent-bg)'
                        : 'rgba(255, 255, 255, 0.05)',
                    border: `1px solid ${isDragging ? 'var(--accent-border)' : 'rgba(224, 64, 251, 0.3)'}`,
                    borderRadius: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: isAddMediaHovered || isDragging
                        ? '0 0 20px var(--accent-glow), inset 0 1px 1px rgba(255, 255, 255, 0.1)'
                        : '0 0 0 transparent',
                    transform: isAddMediaHovered && !isMobile ? 'scale(1.02)' : 'scale(1)',
                    minHeight: isMobile ? '48px' : 'auto',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <input
                    type="file"
                    accept=".mp3"
                    multiple
                    onChange={(e) => onFileUpload(e.target.files)}
                    style={{ display: 'none' }}
                    id="file-input"
                />
                <label
                    htmlFor="file-input"
                    style={{
                        cursor: 'pointer',
                        fontSize: isMobile ? '14px' : '13px',
                        fontWeight: '600',
                        color: isAddMediaHovered ? 'var(--accent)' : 'var(--gray-100)',
                        transition: 'color 0.3s ease',
                    }}
                >
                    + Add Media
                </label>
            </div>
        </div>
    );
}

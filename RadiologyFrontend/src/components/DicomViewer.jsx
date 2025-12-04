import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from 'cornerstone-core';
import * as cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';
import dicomParser from 'dicom-parser';

// Initialize cornerstone WADO Image Loader
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;

const DicomViewer = ({ scanId, onClose }) => {
    const viewerRef = useRef(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [imageLoaded, setImageLoaded] = useState(false);
    const [metadata, setMetadata] = useState(null);
    const [windowLevel, setWindowLevel] = useState({ width: 400, center: 40 });

    useEffect(() => {
        if (!scanId || !viewerRef.current) return;

        const loadDicomImage = async () => {
            try {
                setLoading(true);
                setError(null);

                // Enable the cornerstone element
                const element = viewerRef.current;
                cornerstone.enable(element);

                // Construct the DICOM file URL
                const imageUrl = `wadouri:${import.meta.env.VITE_API_URL || 'http://localhost:3002'}/api/dicom/${scanId}/file`;

                // Load and display the image
                const image = await cornerstone.loadImage(imageUrl);
                cornerstone.displayImage(element, image);

                // Get viewport to extract metadata
                const viewport = cornerstone.getViewport(element);
                setWindowLevel({
                    width: viewport.voi.windowWidth,
                    center: viewport.voi.windowCenter
                });

                // Extract metadata from image
                if (image.data) {
                    const imageMetadata = {
                        rows: image.rows,
                        columns: image.columns,
                        windowCenter: image.windowCenter,
                        windowWidth: image.windowWidth,
                        slope: image.slope,
                        intercept: image.intercept,
                        minPixelValue: image.minPixelValue,
                        maxPixelValue: image.maxPixelValue
                    };
                    setMetadata(imageMetadata);
                }

                setImageLoaded(true);
                setLoading(false);
            } catch (err) {
                console.error('Error loading DICOM image:', err);
                setError(`Failed to load DICOM image: ${err.message}`);
                setLoading(false);
            }
        };

        loadDicomImage();

        // Cleanup
        return () => {
            if (viewerRef.current) {
                try {
                    cornerstone.disable(viewerRef.current);
                } catch (e) {
                    console.warn('Error disabling cornerstone:', e);
                }
            }
        };
    }, [scanId]);

    // Window/Level adjustment
    const adjustWindowLevel = (deltaWidth, deltaCenter) => {
        if (!viewerRef.current || !imageLoaded) return;

        const element = viewerRef.current;
        const viewport = cornerstone.getViewport(element);

        viewport.voi.windowWidth += deltaWidth;
        viewport.voi.windowCenter += deltaCenter;

        setWindowLevel({
            width: viewport.voi.windowWidth,
            center: viewport.voi.windowCenter
        });

        cornerstone.setViewport(element, viewport);
    };

    // Reset viewport
    const resetViewport = () => {
        if (!viewerRef.current || !imageLoaded) return;

        const element = viewerRef.current;
        cornerstone.reset(element);

        const viewport = cornerstone.getViewport(element);
        setWindowLevel({
            width: viewport.voi.windowWidth,
            center: viewport.voi.windowCenter
        });
    };

    // Zoom
    const zoom = (factor) => {
        if (!viewerRef.current || !imageLoaded) return;

        const element = viewerRef.current;
        const viewport = cornerstone.getViewport(element);
        viewport.scale += factor;
        cornerstone.setViewport(element, viewport);
    };

    // Invert colors
    const invertColors = () => {
        if (!viewerRef.current || !imageLoaded) return;

        const element = viewerRef.current;
        const viewport = cornerstone.getViewport(element);
        viewport.invert = !viewport.invert;
        cornerstone.setViewport(element, viewport);
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center">
            <div className="w-full h-full max-w-7xl mx-auto p-4 flex flex-col">
                {/* Header */}
                <div className="flex justify-between items-center mb-4 text-white">
                    <h2 className="text-2xl font-bold">DICOM Viewer - Scan ID: {scanId}</h2>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition"
                    >
                        Close
                    </button>
                </div>

                {/* Viewer Container */}
                <div className="flex-1 flex gap-4">
                    {/* Main Viewer */}
                    <div className="flex-1 bg-black rounded-lg overflow-hidden relative">
                        {loading && (
                            <div className="absolute inset-0 flex items-center justify-center text-white">
                                <div className="text-center">
                                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-white mx-auto mb-4"></div>
                                    <p>Loading DICOM image...</p>
                                </div>
                            </div>
                        )}

                        {error && (
                            <div className="absolute inset-0 flex items-center justify-center text-white">
                                <div className="text-center bg-red-900 bg-opacity-50 p-8 rounded-lg">
                                    <p className="text-xl mb-2">⚠️ Error</p>
                                    <p>{error}</p>
                                </div>
                            </div>
                        )}

                        <div
                            ref={viewerRef}
                            className="w-full h-full"
                            style={{ minHeight: '500px' }}
                        />
                    </div>

                    {/* Controls Panel */}
                    <div className="w-64 bg-gray-800 rounded-lg p-4 text-white space-y-4">
                        <h3 className="text-lg font-bold mb-4">Controls</h3>

                        {/* Window/Level */}
                        <div className="space-y-2">
                            <h4 className="font-semibold">Window/Level</h4>
                            <div className="text-sm space-y-1">
                                <p>Width: {Math.round(windowLevel.width)}</p>
                                <p>Center: {Math.round(windowLevel.center)}</p>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <button
                                    onClick={() => adjustWindowLevel(50, 0)}
                                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    W+
                                </button>
                                <button
                                    onClick={() => adjustWindowLevel(-50, 0)}
                                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    W-
                                </button>
                                <button
                                    onClick={() => adjustWindowLevel(0, 10)}
                                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    L+
                                </button>
                                <button
                                    onClick={() => adjustWindowLevel(0, -10)}
                                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    L-
                                </button>
                            </div>
                        </div>

                        {/* Zoom */}
                        <div className="space-y-2">
                            <h4 className="font-semibold">Zoom</h4>
                            <div className="grid grid-cols-2 gap-2">
                                <button
                                    onClick={() => zoom(0.1)}
                                    className="px-2 py-1 bg-green-600 hover:bg-green-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    Zoom In
                                </button>
                                <button
                                    onClick={() => zoom(-0.1)}
                                    className="px-2 py-1 bg-green-600 hover:bg-green-700 rounded text-xs"
                                    disabled={!imageLoaded}
                                >
                                    Zoom Out
                                </button>
                            </div>
                        </div>

                        {/* Other Controls */}
                        <div className="space-y-2">
                            <h4 className="font-semibold">Tools</h4>
                            <button
                                onClick={invertColors}
                                className="w-full px-2 py-1 bg-purple-600 hover:bg-purple-700 rounded text-xs"
                                disabled={!imageLoaded}
                            >
                                Invert Colors
                            </button>
                            <button
                                onClick={resetViewport}
                                className="w-full px-2 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-xs"
                                disabled={!imageLoaded}
                            >
                                Reset View
                            </button>
                        </div>

                        {/* Metadata */}
                        {metadata && (
                            <div className="space-y-2 pt-4 border-t border-gray-700">
                                <h4 className="font-semibold">Image Info</h4>
                                <div className="text-xs space-y-1">
                                    <p>Size: {metadata.columns} × {metadata.rows}</p>
                                    <p>Min: {metadata.minPixelValue}</p>
                                    <p>Max: {metadata.maxPixelValue}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Instructions */}
                <div className="mt-4 text-white text-sm text-center opacity-75">
                    <p>Use the controls on the right to adjust window/level, zoom, and other settings</p>
                </div>
            </div>
        </div>
    );
};

export default DicomViewer;

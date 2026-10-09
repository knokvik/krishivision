import React, { useState, useRef, useEffect } from 'react'

function UploadCapture({ onImageSelect, preview, onPredict, loading, disabled }) {
  const [cameraActive, setCameraActive] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => {
    if (cameraActive) {
      startCamera()
    } else {
      stopCamera()
    }
    return () => stopCamera()
  }, [cameraActive])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error('Camera access denied:', err)
      setCameraActive(false)
      alert('Camera access denied. Please allow camera permission or use file upload.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  const capturePhoto = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    canvas.toBlob((blob) => {
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' })
      onImageSelect(file)
      setCapturedImage(URL.createObjectURL(blob))
      setCameraActive(false)
    }, 'image/jpeg', 0.9)
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      onImageSelect(file)
    }
  }

  const displayImage = capturedImage || preview

  return (
    <div className="animate-fade-in">
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload or Capture Leaf Image</h2>
          <p className="text-gray-600">Take a clear photo of a single leaf for best results</p>
        </div>

        {cameraActive ? (
          <div className="p-6">
            <div className="relative aspect-video bg-gray-900 rounded-xl overflow-hidden">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
              <canvas ref={canvasRef} className="hidden" />
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-3">
                <button
                  onClick={capturePhoto}
                  disabled={loading}
                  className="bg-white text-gray-900 px-6 py-3 rounded-full font-medium shadow-lg hover:bg-gray-100 transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 2a6 6 0 110 12 6 6 0 010-12z" />
                    <path d="M10 6a4 4 0 110 8 4 4 0 010-8z" />
                  </svg>
                  Capture
                </button>
                <button
                  onClick={() => setCameraActive(false)}
                  className="bg-gray-800 text-white px-6 py-3 rounded-full font-medium shadow-lg hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
            <p className="text-center text-sm text-gray-500 mt-4">Position leaf in frame and tap Capture</p>
          </div>
        ) : displayImage ? (
          <div className="p-6">
            <div className="relative aspect-video bg-gray-100 rounded-xl overflow-hidden">
              <img
                src={displayImage}
                alt="Selected leaf"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 right-3 flex gap-2">
                <button
                  onClick={() => setCameraActive(true)}
                  className="bg-white/90 backdrop-blur-sm text-gray-700 px-3 py-1.5 rounded-lg text-sm font-medium hover:bg-white transition-colors flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Retake
                </button>
                <button
                  onClick={() => onImageSelect(null)}
                  className="bg-white/90 backdrop-blur-sm text-gray-700 px-3 py-1.5 rounded-lg text-sm font-medium hover:bg-white transition-colors"
                >
                  Remove
                </button>
              </div>
            </div>
            <div className="mt-6 text-center">
              <button
                onClick={onPredict}
                disabled={loading || disabled}
                className="bg-primary-600 text-white px-8 py-3 rounded-xl font-semibold text-lg shadow-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mx-auto"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Analyze Leaf
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="p-8">
            <div className="border-2 border-dashed border-gray-300 rounded-2xl p-8 text-center hover:border-primary-400 transition-colors">
              <input
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
                disabled={loading}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <div className="w-16 h-16 bg-primary-50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-gray-600 mb-1">Click or drag to upload a leaf photo</p>
                <p className="text-sm text-gray-400">JPG, PNG up to 10MB</p>
              </label>
            </div>

            <div className="mt-6 text-center">
              <button
                onClick={() => setCameraActive(true)}
                disabled={loading}
                className="bg-white border-2 border-gray-300 text-gray-700 px-8 py-3 rounded-xl font-semibold text-lg shadow-sm hover:bg-gray-50 transition-colors disabled:opacity-50 flex items-center justify-center gap-2 mx-auto"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Use Camera Instead
              </button>
            </div>

            <div className="mt-8 grid grid-cols-3 gap-4 text-sm text-gray-500">
              <div className="flex flex-col items-center gap-1">
                <svg className="w-6 h-6 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Single leaf</span>
              </div>
              <div className="flex flex-col items-center gap-1">
                <svg className="w-6 h-6 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span>Good lighting</span>
              </div>
              <div className="flex flex-col items-center gap-1">
                <svg className="w-6 h-6 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Plain background</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadCapture
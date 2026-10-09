import React, { useState, useCallback } from 'react'
import UploadCapture from './components/UploadCapture'
import ResultCard from './components/ResultCard'
import { predictImage, getRemedy, generateReport } from './api'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [language, setLanguage] = useState('en')
  const [remedy, setRemedy] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'hi', name: 'हिंदी', flag: '🇮🇳' },
    { code: 'mr', name: 'मराठी', flag: '🇮🇳' }
  ]

  const handleImageSelect = useCallback((file) => {
    setImage(file)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    setResult(null)
    setRemedy(null)
    setError(null)
  }, [])

  const handlePredict = async () => {
    if (!image) return
    
    setLoading(true)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', image)
      formData.append('language', language)
      formData.append('include_heatmap', 'true')
      formData.append('include_remedy', 'true')
      
      const response = await predictImage(formData)
      setResult(response)
      
      if (response.remedy) {
        setRemedy(response.remedy)
      } else if (response.verdict === 'accepted') {
        const remedyData = await getRemedy(response.top_3[0].class, response.confidence, language)
        setRemedy(remedyData)
      }
    } catch (err) {
      setError(err.message || 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [image, language])

  const handleLanguageChange = async (lang) => {
    setLanguage(lang)
    if (result && result.verdict === 'accepted' && result.top_3.length > 0) {
      try {
        const remedyData = await getRemedy(result.top_3[0].class, result.confidence, lang)
        setRemedy(remedyData)
      } catch (err) {
        console.error('Failed to fetch remedy in new language:', err)
      }
    }
  }

  const handleDownloadReport = async () => {
    if (!image || !result) return
    
    setReportLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', image)
      formData.append('language', language)
      
      const blob = await generateReport(formData)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `krishivision_report_${result.crop}_${result.disease}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      setError('Failed to generate report. Please try again.')
    } finally {
      setReportLoading(false)
    }
  }

  const handleReset = () => {
    setImage(null)
    setPreview(null)
    setResult(null)
    setRemedy(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">KrishiVision</h1>
                <p className="text-xs text-gray-500">AI Crop Disease Detector</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <select
                value={language}
                onChange={(e) => handleLanguageChange(e.target.value)}
                className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                {languages.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg animate-fade-in flex items-center gap-3">
            <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {!result ? (
          <UploadCapture
            onImageSelect={handleImageSelect}
            preview={preview}
            onPredict={handlePredict}
            loading={loading}
            disabled={!image}
          />
        ) : (
          <div className="animate-slide-up">
            <ResultCard
              result={result}
              remedy={remedy}
              language={language}
              onDownloadReport={handleDownloadReport}
              reportLoading={reportLoading}
              onNewScan={handleReset}
            />
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
          <p>KrishiVision v1.0.0 — AI-powered crop disease detection for farmers</p>
          <p className="mt-1">Not a substitute for professional agricultural advice. Consult local experts for treatment decisions.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
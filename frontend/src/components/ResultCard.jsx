import React from 'react'

function ResultCard({ result, remedy, language, onDownloadReport, reportLoading, onNewScan }) {
  const isHealthy = result.disease.toLowerCase().includes('healthy')
  const confidencePercent = Math.round(result.confidence * 100)
  
  const urgencyColors = {
    none: 'bg-gray-100 text-gray-700',
    low: 'bg-blue-100 text-blue-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800'
  }

  const urgencyLabels = {
    none: 'No Action Needed',
    low: 'Low Priority',
    medium: 'Monitor Closely',
    high: 'Action Required',
    critical: 'Urgent Action'
  }

  if (result.verdict === 'rejected') {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-fade-in">
        <div className="p-6 text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">Image Rejected</h2>
          <p className="text-gray-600 mb-6">{result.reason}</p>
          <button
            onClick={onNewScan}
            className="bg-primary-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors"
          >
            Try Another Image
          </button>
        </div>
      </div>
    )
  }

  if (result.verdict === 'unsure') {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-fade-in">
        <div className="p-6">
          <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-900 text-center mb-2">Uncertain Diagnosis</h2>
          <p className="text-gray-600 text-center mb-4">{result.reason}</p>
          
          <div className="bg-gray-50 rounded-xl p-4 mb-6">
            <h3 className="font-semibold text-gray-900 mb-2">Top Predictions</h3>
            <div className="space-y-2">
              {result.top_3.map((pred, i) => (
                <div key={i} className="flex justify-between items-center p-2 bg-white rounded-lg">
                  <span className="text-sm text-gray-700">{pred.class.replace('___', ' - ')}</span>
                  <span className="font-medium text-gray-900">{Math.round(pred.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={onNewScan}
              className="flex-1 bg-primary-600 text-white py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors"
            >
              Retake Photo
            </button>
            <button
              onClick={onDownloadReport}
              disabled={reportLoading}
              className="flex-1 bg-white border-2 border-gray-300 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              {reportLoading ? 'Generating...' : 'Download Report'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-fade-in">
      <div className={`p-6 ${isHealthy ? 'bg-green-50' : 'bg-red-50'} border-b border-gray-100`}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium text-gray-500">Crop:</span>
              <span className="text-lg font-bold text-gray-900">{result.crop}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-500">Disease:</span>
              <span className={`text-lg font-bold ${isHealthy ? 'text-green-700' : 'text-red-700'}`}>
                {result.disease}
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-900">{confidencePercent}%</div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {result.overlay_b64 && (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Grad-CAM Heatmap</h3>
            <div className="relative aspect-video bg-gray-100 rounded-xl overflow-hidden">
              <img
                src={`data:image/png;base64,${result.overlay_b64}`}
                alt="Grad-CAM overlay"
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-3 left-3 right-3 bg-black/70 text-white text-xs p-2 rounded">
                Red/Yellow regions indicate areas that most influenced the diagnosis
              </div>
            </div>
          </div>
        )}

        {remedy && (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Treatment Recommendations</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl p-4">
                <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Organic Treatment
                </h4>
                <p className="text-gray-700 text-sm leading-relaxed">{remedy.organic_treatment}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4">
                <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Chemical Treatment
                </h4>
                <p className="text-gray-700 text-sm leading-relaxed">{remedy.chemical_treatment}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4">
                <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                  Prevention
                </h4>
                <p className="text-gray-700 text-sm leading-relaxed">{remedy.prevention}</p>
              </div>
              <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-xl">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${urgencyColors[remedy.urgency] || urgencyColors.medium}`}>
                  {urgencyLabels[remedy.urgency] || remedy.urgency.toUpperCase()}
                </span>
                <span className="text-sm text-gray-600">{remedy.severity}</span>
              </div>
            </div>
          </div>
        )}

        <div className="bg-gray-50 rounded-xl p-4">
          <h4 className="font-medium text-gray-900 mb-2">Top 3 Predictions</h4>
          <div className="space-y-2">
            {result.top_3.map((pred, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-white rounded-lg">
                <div className="flex items-center gap-3">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                    i === 0 ? 'bg-primary-100 text-primary-700' : 'bg-gray-100 text-gray-600'
                  }`}>
                    {i + 1}
                  </span>
                  <span className="text-sm text-gray-700">{pred.class.replace('___', ' - ')}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        i === 0 ? 'bg-primary-500' : 'bg-gray-400'
                      }`}
                      style={{ width: `${Math.round(pred.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="font-medium text-gray-900 w-12 text-right">
                    {Math.round(pred.confidence * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t border-gray-100 flex flex-col sm:flex-row gap-3">
          <button
            onClick={onDownloadReport}
            disabled={reportLoading}
            className="flex-1 bg-primary-600 text-white py-3 rounded-xl font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            {reportLoading ? 'Generating PDF...' : 'Download PDF Report'}
          </button>
          <button
            onClick={onNewScan}
            className="flex-1 bg-white border-2 border-gray-300 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-50 transition-colors"
          >
            New Scan
          </button>
        </div>
      </div>
    </div>
  )
}

export default ResultCard
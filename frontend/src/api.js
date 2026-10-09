import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

export async function predictImage(formData) {
  const response = await api.post('/predict', formData)
  return response.data
}

export async function getRemedy(className, confidence, language = 'en') {
  const formData = new FormData()
  formData.append('class_name', className)
  formData.append('confidence', confidence.toString())
  
  const response = await api.post(`/remedy?lang=${language}`, formData)
  return response.data
}

export async function generateReport(formData) {
  const response = await api.post('/report', formData, {
    responseType: 'blob'
  })
  return response.data
}

export async function healthCheck() {
  const response = await api.get('/health')
  return response.data
}

export default api
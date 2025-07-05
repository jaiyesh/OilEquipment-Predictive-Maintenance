import json
import pandas as pd
from datetime import datetime
import openai  # You can also use other LLM APIs
from typing import Dict, List, Any
import numpy as np
import os

# For RAG implementation (you can use alternatives like Langchain, llamaindex)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import PyPDF2
    RAG_AVAILABLE = True
except ImportError:
    print("RAG dependencies not available. Install sentence-transformers and PyPDF2 for full functionality.")
    RAG_AVAILABLE = False

class PumpMaintenanceRAG:
    def __init__(self, pdf_path=None, model_name="all-MiniLM-L6-v2"):
        """Initialize RAG system with pump manual"""
        self.pdf_path = pdf_path
        self.knowledge_base = []
        self.embeddings = []
        self.embedding_model = None
        
        if RAG_AVAILABLE:
            self.embedding_model = SentenceTransformer(model_name)
            if pdf_path:
                self.load_manual(pdf_path)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF manual"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return self.get_manual_text()  # Fallback to hardcoded manual
    
    def get_manual_text(self):
        """Fallback manual text (subset of the full manual)"""
        return """
        CENTRIFUGAL PUMP MODEL CP-5000 TROUBLESHOOTING GUIDE
        
        BEARING WEAR INDICATORS:
        - Bearing temperature increase >5°F above baseline
        - Vibration amplitude increase >20% in high frequency range
        - Oil analysis showing metallic particles
        - Recommended Actions: Stage 1 - Increase monitoring frequency, Stage 2 - Schedule bearing replacement
        
        IMPELLER DAMAGE INDICATORS:
        - Discharge pressure drop 5-15%
        - Flow rate reduction with same valve position
        - Motor current increase 10-20%
        - Recommended Actions: Schedule impeller inspection/replacement
        
        MOTOR OVERHEATING INDICATORS:
        - Motor temperature increase 10-15°F above normal
        - Current imbalance >5% between phases
        - Speed reduction under load
        - Recommended Actions: Check electrical connections, plan motor maintenance
        
        SEAL DEGRADATION INDICATORS:
        - Leak rate increase from <2 to 3-5 ml/hr
        - Visible process fluid leakage
        - Recommended Actions: Monitor trends, plan seal replacement
        
        CAVITATION INDICATORS:
        - Suction pressure drops below 10 PSIG
        - High frequency vibration >2000 Hz
        - Audible cavitation noise
        - Recommended Actions: Increase suction pressure, reduce flow
        
        NORMAL OPERATING RANGES:
        - Discharge pressure: 85-95 PSIG
        - Suction pressure: 15-25 PSIG
        - Flow rate: 450-550 GPM
        - Motor current: 180-220 Amps
        - Bearing temperature: 70-85°F
        - Motor temperature: 140-160°F
        - Vibration levels: <0.15 in/sec
        """
    
    def load_manual(self, pdf_path):
        """Load and process pump manual for RAG"""
        print("Loading pump manual...")
        
        if pdf_path.endswith('.pdf'):
            manual_text = self.extract_text_from_pdf(pdf_path)
        else:
            manual_text = self.get_manual_text()
        
        # Split text into chunks for better retrieval
        chunks = self.split_text_into_chunks(manual_text, chunk_size=500, overlap=50)
        self.knowledge_base = chunks
        
        # Generate embeddings
        if self.embedding_model:
            print("Generating embeddings...")
            self.embeddings = self.embedding_model.encode(chunks)
            print(f"Loaded {len(chunks)} text chunks into knowledge base")
    
    def split_text_into_chunks(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant manual sections"""
        if not self.embeddings or not RAG_AVAILABLE:
            return [self.get_manual_text()[:1000]]  # Return subset as fallback
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [self.knowledge_base[i] for i in top_indices]
        
        return relevant_chunks

class MaintenanceReportGenerator:
    def __init__(self, rag_system=None, llm_api_key=None):
        """Initialize report generator with RAG and LLM"""
        self.rag_system = rag_system or PumpMaintenanceRAG()
        self.llm_api_key = llm_api_key
        
        # Initialize OpenAI client (you can substitute with other LLM providers)
        if llm_api_key:
            openai.api_key = llm_api_key
    
    def analyze_sensor_data(self, sensor_readings):
        """Analyze current sensor readings against normal ranges"""
        analysis = {
            'normal_parameters': [],
            'warning_parameters': [],
            'critical_parameters': [],
            'trends': []
        }
        
        # Define normal ranges (from manual)
        normal_ranges = {
            'discharge_pressure': (85, 95),
            'suction_pressure': (15, 25),
            'flow_rate': (450, 550),
            'motor_current': (180, 220),
            'bearing_temp': (70, 85),
            'motor_temp': (140, 160),
            'vibration_magnitude': (0.05, 0.15),
            'seal_leak_rate': (0, 2)
        }
        
        for param, value in sensor_readings.items():
            if param in normal_ranges:
                min_val, max_val = normal_ranges[param]
                
                if min_val <= value <= max_val:
                    analysis['normal_parameters'].append(param)
                elif value > max_val * 1.1 or value < min_val * 0.9:
                    analysis['critical_parameters'].append({
                        'parameter': param,
                        'value': value,
                        'normal_range': normal_ranges[param],
                        'deviation': ((value - max_val) / max_val * 100) if value > max_val else ((min_val - value) / min_val * 100)
                    })
                else:
                    analysis['warning_parameters'].append({
                        'parameter': param,
                        'value': value,
                        'normal_range': normal_ranges[param]
                    })
        
        return analysis
    
    def determine_failure_mode(self, analysis_data):
        """Determine potential failure modes based on sensor patterns"""
        critical_params = [item['parameter'] for item in analysis_data['critical_parameters']]
        warning_params = [item['parameter'] for item in analysis_data['warning_parameters']]
        
        failure_modes = []
        
        # Bearing wear detection
        if 'bearing_temp' in critical_params or 'vibration_magnitude' in critical_params:
            failure_modes.append({
                'type': 'bearing_wear',
                'confidence': 0.8,
                'description': 'Possible bearing wear indicated by temperature and/or vibration increase'
            })
        
        # Motor issues detection
        if 'motor_temp' in critical_params or 'motor_current' in critical_params:
            failure_modes.append({
                'type': 'motor_issues',
                'confidence': 0.7,
                'description': 'Motor problems indicated by temperature or current anomalies'
            })
        
        # Seal degradation detection
        if 'seal_leak_rate' in critical_params:
            failure_modes.append({
                'type': 'seal_degradation',
                'confidence': 0.9,
                'description': 'Mechanical seal degradation indicated by increased leak rate'
            })
        
        # Cavitation detection
        if 'suction_pressure' in critical_params and 'vibration_magnitude' in critical_params:
            failure_modes.append({
                'type': 'cavitation',
                'confidence': 0.8,
                'description': 'Cavitation indicated by low suction pressure and high vibration'
            })
        
        return failure_modes
    
    def generate_recommendations(self, failure_modes, analysis_data):
        """Generate specific maintenance recommendations"""
        recommendations = []
        
        for failure_mode in failure_modes:
            if failure_mode['type'] == 'bearing_wear':
                recommendations.extend([
                    {
                        'priority': 'HIGH',
                        'action': 'Schedule bearing inspection within 48 hours',
                        'reason': 'Elevated bearing temperature/vibration detected',
                        'estimated_cost': '$800-1200',
                        'downtime': '4-6 hours'
                    },
                    {
                        'priority': 'MEDIUM',
                        'action': 'Increase vibration monitoring frequency to every 4 hours',
                        'reason': 'Track bearing condition deterioration rate',
                        'estimated_cost': '$0',
                        'downtime': '0 hours'
                    }
                ])
            
            elif failure_mode['type'] == 'motor_issues':
                recommendations.extend([
                    {
                        'priority': 'HIGH',
                        'action': 'Check motor electrical connections and phase balance',
                        'reason': 'Motor temperature/current anomalies detected',
                        'estimated_cost': '$200-500',
                        'downtime': '2-3 hours'
                    },
                    {
                        'priority': 'MEDIUM',
                        'action': 'Schedule motor insulation resistance test',
                        'reason': 'Verify motor winding condition',
                        'estimated_cost': '$300-400',
                        'downtime': '1-2 hours'
                    }
                ])
            
            elif failure_mode['type'] == 'seal_degradation':
                recommendations.extend([
                    {
                        'priority': 'HIGH',
                        'action': 'Plan mechanical seal replacement within 2 weeks',
                        'reason': 'Seal leak rate exceeds acceptable limits',
                        'estimated_cost': '$2500-3000',
                        'downtime': '8-12 hours'
                    },
                    {
                        'priority': 'IMMEDIATE',
                        'action': 'Check seal flush system operation',
                        'reason': 'Ensure proper seal cooling and lubrication',
                        'estimated_cost': '$100-200',
                        'downtime': '1 hour'
                    }
                ])
            
            elif failure_mode['type'] == 'cavitation':
                recommendations.extend([
                    {
                        'priority': 'IMMEDIATE',
                        'action': 'Increase suction pressure or reduce flow rate',
                        'reason': 'Prevent impeller damage from cavitation',
                        'estimated_cost': '$0',
                        'downtime': '0 hours'
                    },
                    {
                        'priority': 'HIGH',
                        'action': 'Inspect suction piping for blockages',
                        'reason': 'Eliminate sources of pressure drop',
                        'estimated_cost': '$500-800',
                        'downtime': '2-4 hours'
                    }
                ])
        
        # Add general recommendations if no specific failure modes detected
        if not failure_modes:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Continue normal monitoring schedule',
                'reason': 'All parameters within acceptable ranges',
                'estimated_cost': '$0',
                'downtime': '0 hours'
            })
        
        return recommendations
    
    def create_llm_prompt(self, report_data, relevant_context):
        """Create prompt for LLM report generation"""
        prompt = f"""
        You are an expert maintenance engineer for oil and gas equipment. Generate a comprehensive maintenance report for a centrifugal pump based on the following data:

        EQUIPMENT: Centrifugal Pump Model CP-5000, ID: {report_data.get('equipment_id', 'Unknown')}
        TIMESTAMP: {report_data.get('analysis_timestamp', 'Unknown')}
        OVERALL STATUS: {report_data.get('overall_status', 'Unknown')}

        CURRENT SENSOR READINGS:
        {json.dumps(report_data.get('sensor_readings', {}), indent=2)}

        ML ANOMALY DETECTION:
        - Anomaly Detected: {report_data.get('anomaly_detection', {}).get('ml_prediction', False)}
        - Confidence Score: {report_data.get('anomaly_detection', {}).get('confidence_score', 0)}
        - Rule Violations: {report_data.get('anomaly_detection', {}).get('rule_violations', [])}

        TREND ANALYSIS:
        {json.dumps(report_data.get('trends', {}), indent=2)}

        RELEVANT MANUAL SECTIONS:
        {chr(10).join(relevant_context)}

        Please generate a professional maintenance report that includes:
        1. Executive Summary
        2. Current Equipment Status
        3. Anomaly Analysis
        4. Risk Assessment
        5. Recommended Actions with Priorities
        6. Cost and Downtime Estimates

        Format the report professionally and provide specific, actionable recommendations.
        """
        
        return prompt
    
    def generate_report_with_llm(self, report_data):
        """Generate report using LLM with RAG context"""
        try:
            # Create query for RAG retrieval
            query = f"troubleshooting {report_data.get('overall_status', '')} pump issues sensor readings anomalies"
            
            # Retrieve relevant context from manual
            relevant_context = self.rag_system.retrieve_relevant_context(query, top_k=3)
            
            # Create LLM prompt
            prompt = self.create_llm_prompt(report_data, relevant_context)
            
            # Generate report using LLM (OpenAI example)
            if self.llm_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert maintenance engineer specializing in oil and gas equipment."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                return response.choices[0].message.content
            else:
                # Fallback to rule-based report if no LLM API
                return self.generate_fallback_report(report_data)
                
        except Exception as e:
            print(f"Error generating LLM report: {e}")
            return self.generate_fallback_report(report_data)
    
    def generate_fallback_report(self, report_data):
        """Generate rule-based report as fallback"""
        sensor_analysis = self.analyze_sensor_data(report_data.get('sensor_readings', {}))
        failure_modes = self.determine_failure_mode(sensor_analysis)
        recommendations = self.generate_recommendations(failure_modes, sensor_analysis)
        
        report = f"""
        MAINTENANCE REPORT - CENTRIFUGAL PUMP CP-5000
        ============================================
        
        Equipment ID: {report_data.get('equipment_id', 'Unknown')}
        Analysis Time: {report_data.get('analysis_timestamp', 'Unknown')}
        Overall Status: {report_data.get('overall_status', 'Unknown')}
        
        EXECUTIVE SUMMARY:
        {self._generate_executive_summary(report_data, failure_modes)}
        
        CURRENT EQUIPMENT STATUS:
        - Normal Parameters: {len(sensor_analysis['normal_parameters'])}
        - Warning Parameters: {len(sensor_analysis['warning_parameters'])}
        - Critical Parameters: {len(sensor_analysis['critical_parameters'])}
        
        ANOMALY ANALYSIS:
        ML Anomaly Detection: {'YES' if report_data.get('anomaly_detection', {}).get('ml_prediction') else 'NO'}
        Confidence Score: {report_data.get('anomaly_detection', {}).get('confidence_score', 0):.2f}
        
        IDENTIFIED ISSUES:
        {self._format_critical_parameters(sensor_analysis['critical_parameters'])}
        
        POTENTIAL FAILURE MODES:
        {self._format_failure_modes(failure_modes)}
        
        RECOMMENDED ACTIONS:
        {self._format_recommendations(recommendations)}
        
        RISK ASSESSMENT:
        {self._generate_risk_assessment(failure_modes, sensor_analysis)}
        
        ============================================
        Report generated by Predictive Maintenance System
        """
        
        return report
    
    def _generate_executive_summary(self, report_data, failure_modes):
        """Generate executive summary"""
        if report_data.get('overall_status') == 'NORMAL':
            return "Equipment is operating within normal parameters. Continue routine monitoring."
        elif failure_modes:
            return f"ATTENTION REQUIRED: {len(failure_modes)} potential issue(s) detected requiring maintenance action."
        else:
            return "Equipment shows some anomalies but no specific failure modes identified. Increased monitoring recommended."
    
    def _format_critical_parameters(self, critical_params):
        """Format critical parameters section"""
        if not critical_params:
            return "No critical parameter violations detected."
        
        formatted = ""
        for param in critical_params:
            formatted += f"- {param['parameter']}: {param['value']:.2f} (Normal: {param['normal_range']}, Deviation: {param['deviation']:.1f}%)\n"
        
        return formatted
    
    def _format_failure_modes(self, failure_modes):
        """Format failure modes section"""
        if not failure_modes:
            return "No specific failure modes identified."
        
        formatted = ""
        for mode in failure_modes:
            formatted += f"- {mode['type'].replace('_', ' ').title()}: {mode['description']} (Confidence: {mode['confidence']*100:.0f}%)\n"
        
        return formatted
    
    def _format_recommendations(self, recommendations):
        """Format recommendations section"""
        if not recommendations:
            return "No specific recommendations at this time."
        
        formatted = ""
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. [{rec['priority']}] {rec['action']}\n"
            formatted += f"   Reason: {rec['reason']}\n"
            formatted += f"   Cost: {rec['estimated_cost']}, Downtime: {rec['downtime']}\n\n"
        
        return formatted
    
    def _generate_risk_assessment(self, failure_modes, sensor_analysis):
        """Generate risk assessment"""
        critical_count = len(sensor_analysis['critical_parameters'])
        failure_mode_count = len(failure_modes)
        
        if critical_count == 0 and failure_mode_count == 0:
            return "LOW RISK: Equipment operating normally."
        elif critical_count <= 2 and failure_mode_count <= 1:
            return "MEDIUM RISK: Monitor closely and plan maintenance."
        else:
            return "HIGH RISK: Immediate attention required to prevent equipment failure."

# Complete system integration
class PredictiveMaintenanceSystem:
    def __init__(self, model_path=None, manual_path=None, llm_api_key=None):
        """Initialize complete predictive maintenance system"""
        # Import the monitoring system from ml_anomaly_detection module
        try:
            # Try to import from the ml_anomaly_detection module
            import importlib.util
            spec = importlib.util.spec_from_file_location("ml_anomaly_detection", "ml_anomaly_detection.py")
            ml_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ml_module)
            PumpMonitoringSystem = ml_module.PumpMonitoringSystem
        except:
            # Fallback: assume the class is already available in globals
            # This will work when the ML module is executed first
            pass
        
        self.monitoring_system = PumpMonitoringSystem(model_path)
        self.rag_system = PumpMaintenanceRAG(manual_path)
        self.report_generator = MaintenanceReportGenerator(self.rag_system, llm_api_key)
    
    def process_real_time_data(self, sensor_data):
        """Complete pipeline: monitoring -> analysis -> report generation"""
        # Step 1: ML-based anomaly detection
        monitoring_results = self.monitoring_system.real_time_monitoring(sensor_data)
        
        # Step 2: Generate structured report data
        report_data = self.monitoring_system.generate_maintenance_report(
            monitoring_results, sensor_data
        )
        
        # Step 3: Generate comprehensive report with LLM
        final_report = self.report_generator.generate_report_with_llm(report_data)
        
        return {
            'monitoring_results': monitoring_results,
            'report_data': report_data,
            'maintenance_report': final_report
        }

# Usage example
if __name__ == "__main__":
    # Initialize the complete system
    system = PredictiveMaintenanceSystem(
        model_path='pump_anomaly_model.pkl',
        manual_path='pump_manual.pdf',  # Or None to use fallback text
        llm_api_key=os.getenv('OPENAI_API_KEY')  # Add your OpenAI API key here
    )
    
    # Example sensor data
    example_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'discharge_pressure': [92.5],
        'suction_pressure': [18.2],
        'flow_rate': [485.0],
        'motor_current': [205.0],
        'bearing_temp': [88.5],  # Slightly elevated
        'motor_temp': [155.0],
        'vibration_x': [0.12],  # Individual vibration components
        'vibration_y': [0.11],
        'vibration_z': [0.09],
        'motor_rpm': [3575.0],
        'oil_pressure': [29.0],
        'seal_leak_rate': [1.5]
    })
    
    # Process data through complete pipeline
    results = system.process_real_time_data(example_data)
    
    print("MAINTENANCE REPORT:")
    print("=" * 50)
    print(results['maintenance_report'])

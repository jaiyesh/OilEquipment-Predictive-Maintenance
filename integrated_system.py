#!/usr/bin/env python3
"""
Integrated Predictive Maintenance System
This script combines all components and handles imports cleanly
"""

import os
import sys
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all required components
def load_all_modules():
    """Load all system modules"""
    global CentrifugalPumpDataGenerator, PumpAnomalyDetector, PumpMonitoringSystem
    global PumpMaintenanceRAG, MaintenanceReportGenerator
    
    print("Loading system modules...")
    
    # Execute pump data generator
    with open('pump_data_generator.py', 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
    
    # Execute ML anomaly detection
    with open('ml_anomaly_detection.py', 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
    
    # Execute LLM report generator
    with open('llm_report_generator.py', 'r', encoding='utf-8') as f:
        exec(f.read(), globals())
    
    print("✅ All modules loaded successfully!")

class IntegratedMaintenanceSystem:
    """Complete integrated predictive maintenance system"""
    
    def __init__(self):
        self.data_file = 'centrifugal_pump_sensor_data.csv'
        self.model_file = 'pump_anomaly_model.pkl'
        self.manual_file = 'pump_manual.pdf'
        
        # System components
        self.data_generator = None
        self.monitoring_system = None
        self.rag_system = None
        self.report_generator = None
    
    def generate_training_data(self, days=365, failure_modes=3):
        """Generate synthetic training data"""
        print(f"\n🔄 Generating {days} days of training data...")
        
        self.data_generator = CentrifugalPumpDataGenerator(
            start_date='2023-01-01',
            sampling_interval_minutes=15
        )
        
        # Generate data
        pump_data = self.data_generator.generate_complete_dataset(
            duration_days=days,
            num_failure_modes=failure_modes
        )
        
        # Save data
        self.data_generator.save_dataset(pump_data, self.data_file)
        
        print(f"✅ Training data generated: {pump_data.shape}")
        return pump_data
    
    def train_anomaly_model(self, model_type='random_forest'):
        """Train the anomaly detection model"""
        print(f"\n🤖 Training {model_type} anomaly detection model...")
        
        # Load data
        if not os.path.exists(self.data_file):
            print("❌ Training data not found. Generating data first...")
            self.generate_training_data()
        
        df = pd.read_csv(self.data_file)
        print(f"📊 Loaded {len(df)} training records")
        
        # Initialize and train detector
        detector = PumpAnomalyDetector(model_type=model_type)
        results = detector.train_model(df, test_size=0.2)
        
        # Save model
        detector.save_model(self.model_file)
        print(f"✅ Model trained and saved: {self.model_file}")
        
        return detector
    
    def setup_monitoring_system(self):
        """Setup the monitoring system with trained model"""
        print("\n📡 Setting up monitoring system...")
        
        if not os.path.exists(self.model_file):
            print("❌ Trained model not found. Training model first...")
            self.train_anomaly_model()
        
        # Initialize monitoring system
        self.monitoring_system = PumpMonitoringSystem(self.model_file)
        
        # Setup RAG system
        manual_path = self.manual_file if os.path.exists(self.manual_file) else None
        self.rag_system = PumpMaintenanceRAG(pdf_path=manual_path)
        
        # Setup report generator
        self.report_generator = MaintenanceReportGenerator(
            self.rag_system, 
            llm_api_key=None  # Set to your OpenAI key for LLM features
        )
        
        print("✅ Monitoring system ready!")
        return True
    
    def analyze_real_time_data(self, sensor_data):
        """Analyze real-time sensor data and generate report"""
        if not self.monitoring_system:
            print("❌ Monitoring system not initialized!")
            return None
        
        # Step 1: ML anomaly detection
        monitoring_results = self.monitoring_system.real_time_monitoring(sensor_data)
        
        # Step 2: Generate report data
        report_data = self.monitoring_system.generate_maintenance_report(
            monitoring_results, sensor_data
        )
        
        # Step 3: Generate comprehensive report
        maintenance_report = self.report_generator.generate_fallback_report(report_data)
        
        return {
            'monitoring_results': monitoring_results,
            'report_data': report_data,
            'maintenance_report': maintenance_report,
            'timestamp': datetime.now()
        }
    
    def run_test_scenarios(self):
        """Run predefined test scenarios"""
        print("\n🧪 Running test scenarios...")
        
        test_scenarios = [
            {
                'name': 'Normal Operation',
                'data': pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'discharge_pressure': [90.0],
                    'suction_pressure': [20.0], 
                    'flow_rate': [500.0],
                    'motor_current': [200.0],
                    'bearing_temp': [75.0],
                    'motor_temp': [150.0],
                    'vibration_x': [0.10],
                    'vibration_y': [0.10],
                    'vibration_z': [0.08],
                    'motor_rpm': [3580.0],
                    'oil_pressure': [30.0],
                    'seal_leak_rate': [1.0]
                })
            },
            {
                'name': 'Bearing Overheating',
                'data': pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'discharge_pressure': [88.0],
                    'suction_pressure': [19.0],
                    'flow_rate': [480.0],
                    'motor_current': [220.0],
                    'bearing_temp': [95.0],  # HIGH
                    'motor_temp': [155.0],
                    'vibration_x': [0.25],   # HIGH
                    'vibration_y': [0.23],   # HIGH
                    'vibration_z': [0.18],
                    'motor_rpm': [3570.0],
                    'oil_pressure': [28.0],
                    'seal_leak_rate': [1.5]
                })
            },
            {
                'name': 'Seal Failure',
                'data': pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'discharge_pressure': [85.0],
                    'suction_pressure': [18.0],
                    'flow_rate': [490.0],
                    'motor_current': [205.0],
                    'bearing_temp': [80.0],
                    'motor_temp': [152.0],
                    'vibration_x': [0.12],
                    'vibration_y': [0.11],
                    'vibration_z': [0.09],
                    'motor_rpm': [3575.0],
                    'oil_pressure': [29.0],
                    'seal_leak_rate': [12.0]  # CRITICAL
                })
            }
        ]
        
        results = []
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Scenario {i}: {scenario['name']} ---")
            
            # Analyze scenario
            result = self.analyze_real_time_data(scenario['data'])
            results.append(result)
            
            # Display summary
            monitoring = result['monitoring_results']
            print(f"🔍 Status: {monitoring['overall_status']}")
            print(f"🤖 ML Anomaly: {'YES' if monitoring['ml_anomaly_detected'] else 'NO'}")
            print(f"📊 Score: {monitoring['anomaly_score']:.3f}")
            print(f"⚠️  Alerts: {len(monitoring['rule_based_alerts'])}")
            
            # Save detailed report
            report_file = f"test_scenario_{i}_{scenario['name'].replace(' ', '_').lower()}.txt"
            with open(report_file, 'w') as f:
                f.write(result['maintenance_report'])
            print(f"📄 Report saved: {report_file}")
        
        print(f"\n✅ Completed {len(test_scenarios)} test scenarios!")
        return results
    
    def simulate_continuous_monitoring(self, hours=24):
        """Simulate continuous monitoring for specified hours"""
        print(f"\n🔄 Simulating {hours} hours of continuous monitoring...")
        
        # Load historical data for simulation
        if not os.path.exists(self.data_file):
            print("❌ Historical data not found. Generating data first...")
            self.generate_training_data()
        
        historical_data = pd.read_csv(self.data_file)
        
        # Take recent data for simulation
        samples_per_hour = 4  # 15-minute intervals
        total_samples = hours * samples_per_hour
        recent_data = historical_data.tail(total_samples).copy()
        
        # Update timestamps to current time
        recent_data['timestamp'] = pd.date_range(
            start=datetime.now().replace(minute=0, second=0, microsecond=0),
            periods=len(recent_data),
            freq='15min'
        )
        
        # Monitor in hourly batches
        anomaly_count = 0
        critical_alerts = []
        
        for hour in range(hours):
            start_idx = hour * samples_per_hour
            end_idx = start_idx + samples_per_hour
            hour_data = recent_data.iloc[start_idx:end_idx]
            
            # Analyze this hour's data
            result = self.analyze_real_time_data(hour_data)
            monitoring = result['monitoring_results']
            
            # Check for anomalies
            if monitoring['overall_status'] != 'NORMAL':
                anomaly_count += 1
                timestamp = hour_data['timestamp'].iloc[-1]
                
                print(f"⚠️  Hour {hour+1:2d} ({timestamp.strftime('%H:%M')}): {monitoring['overall_status']}")
                
                # Save critical alerts
                if monitoring['anomaly_score'] > 0.7:
                    critical_file = f"CRITICAL_ALERT_hour_{hour+1:02d}_{timestamp.strftime('%H%M')}.txt"
                    with open(critical_file, 'w') as f:
                        f.write(result['maintenance_report'])
                    critical_alerts.append(critical_file)
                    print(f"   🚨 Critical alert saved: {critical_file}")
            else:
                print(f"✅ Hour {hour+1:2d}: Normal operation")
        
        # Summary
        print(f"\n📈 MONITORING SUMMARY ({hours} hours):")
        print(f"   Total hours monitored: {hours}")
        print(f"   Anomalies detected: {anomaly_count}")
        print(f"   Critical alerts: {len(critical_alerts)}")
        print(f"   System availability: {((hours-anomaly_count)/hours)*100:.1f}%")
        
        return {
            'hours_monitored': hours,
            'anomalies_detected': anomaly_count,
            'critical_alerts': critical_alerts
        }

def main():
    """Main execution function"""
    print("🏭 INTEGRATED PREDICTIVE MAINTENANCE SYSTEM")
    print("🔧 Centrifugal Pump Model CP-5000")
    print("=" * 60)
    
    try:
        # Load all system modules
        load_all_modules()
        
        # Initialize integrated system
        system = IntegratedMaintenanceSystem()
        
        print("\n🚀 Choose execution mode:")
        print("1. Complete setup and testing (recommended)")
        print("2. Quick test with existing model")
        print("3. Continuous monitoring simulation")
        print("4. Custom scenario testing")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🎯 Running complete setup and testing...")
            
            # Step 1: Generate data
            system.generate_training_data(days=365, failure_modes=3)
            
            # Step 2: Train model
            system.train_anomaly_model(model_type='random_forest')
            
            # Step 3: Setup monitoring
            system.setup_monitoring_system()
            
            # Step 4: Test scenarios
            system.run_test_scenarios()
            
            # Step 5: Continuous monitoring
            system.simulate_continuous_monitoring(hours=12)
            
            print("\n🎉 Complete setup and testing finished!")
            
        elif choice == "2":
            print("\n⚡ Quick test mode...")
            system.setup_monitoring_system()
            system.run_test_scenarios()
            
        elif choice == "3":
            print("\n🔄 Continuous monitoring simulation...")
            hours = int(input("Enter hours to simulate (default 24): ") or "24")
            system.setup_monitoring_system()
            system.simulate_continuous_monitoring(hours=hours)
            
        elif choice == "4":
            print("\n🧪 Custom scenario testing...")
            system.setup_monitoring_system()
            
            # Get custom sensor values
            print("\nEnter sensor values (press Enter for defaults):")
            
            custom_data = pd.DataFrame({
                'timestamp': [datetime.now()],
                'discharge_pressure': [float(input("Discharge pressure (85-95 PSIG, default 90): ") or "90")],
                'suction_pressure': [float(input("Suction pressure (15-25 PSIG, default 20): ") or "20")],
                'flow_rate': [float(input("Flow rate (450-550 GPM, default 500): ") or "500")],
                'motor_current': [float(input("Motor current (180-220 Amps, default 200): ") or "200")],
                'bearing_temp': [float(input("Bearing temp (70-85°F, default 75): ") or "75")],
                'motor_temp': [float(input("Motor temp (140-160°F, default 150): ") or "150")],
                'vibration_x': [float(input("Vibration X (0.05-0.15 in/sec, default 0.10): ") or "0.10")],
                'vibration_y': [float(input("Vibration Y (0.05-0.15 in/sec, default 0.10): ") or "0.10")],
                'vibration_z': [float(input("Vibration Z (0.03-0.12 in/sec, default 0.08): ") or "0.08")],
                'motor_rpm': [float(input("Motor RPM (3550-3600, default 3580): ") or "3580")],
                'oil_pressure': [float(input("Oil pressure (25-35 PSIG, default 30): ") or "30")],
                'seal_leak_rate': [float(input("Seal leak rate (0-2 ml/hr, default 1): ") or "1")]
            })
            
            result = system.analyze_real_time_data(custom_data)
            
            print(f"\n📊 ANALYSIS RESULTS:")
            monitoring = result['monitoring_results']
            print(f"Status: {monitoring['overall_status']}")
            print(f"ML Anomaly: {'YES' if monitoring['ml_anomaly_detected'] else 'NO'}")
            print(f"Anomaly Score: {monitoring['anomaly_score']:.3f}")
            print(f"Rule Alerts: {len(monitoring['rule_based_alerts'])}")
            
            # Save report
            with open("custom_scenario_report.txt", 'w') as f:
                f.write(result['maintenance_report'])
            print("\n📄 Detailed report saved: custom_scenario_report.txt")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 Execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
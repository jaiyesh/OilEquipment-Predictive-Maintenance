import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CentrifugalPumpDataGenerator:
    def __init__(self, start_date='2023-01-01', sampling_interval_minutes=15):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.sampling_interval = timedelta(minutes=sampling_interval_minutes)
        
        # Operating parameters (normal ranges)
        self.normal_ranges = {
            'discharge_pressure': (85, 95),  # psi
            'suction_pressure': (15, 25),    # psi
            'flow_rate': (450, 550),         # gpm
            'motor_current': (180, 220),     # amps
            'bearing_temp': (70, 85),        # °F
            'motor_temp': (140, 160),        # °F
            'vibration_x': (0.05, 0.15),    # in/sec
            'vibration_y': (0.05, 0.15),    # in/sec
            'vibration_z': (0.03, 0.12),    # in/sec
            'motor_rpm': (3550, 3600),      # rpm
            'oil_pressure': (25, 35),       # psi
            'seal_leak_rate': (0, 2),       # ml/hr
        }
        
        # Failure mode signatures
        self.failure_modes = {
            'bearing_wear': {
                'bearing_temp': 1.5,
                'vibration_x': 2.0,
                'vibration_y': 2.0,
                'motor_current': 1.1
            },
            'impeller_damage': {
                'discharge_pressure': 0.7,
                'flow_rate': 0.8,
                'vibration_x': 1.8,
                'motor_current': 1.15
            },
            'seal_degradation': {
                'seal_leak_rate': 8.0,
                'bearing_temp': 1.2,
                'discharge_pressure': 0.95
            },
            'motor_issues': {
                'motor_temp': 1.4,
                'motor_current': 1.25,
                'motor_rpm': 0.98,
                'vibration_z': 1.6
            },
            'cavitation': {
                'suction_pressure': 0.6,
                'vibration_x': 3.0,
                'vibration_y': 2.5,
                'flow_rate': 0.85,
                'discharge_pressure': 0.9
            }
        }

    def generate_baseline_data(self, duration_days=365):
        """Generate normal operating data"""
        num_samples = int(duration_days * 24 * 60 / self.sampling_interval.total_seconds() * 60)
        
        data = {
            'timestamp': [self.start_date + i * self.sampling_interval for i in range(num_samples)],
            'equipment_id': ['PUMP_001'] * num_samples,
            'status': ['Normal'] * num_samples
        }
        
        # Generate base sensor readings with realistic correlations
        for param, (min_val, max_val) in self.normal_ranges.items():
            if param == 'motor_rpm':
                # RPM should be more stable
                base_value = np.mean([min_val, max_val])
                data[param] = np.random.normal(base_value, (max_val - min_val) * 0.02, num_samples)
            elif 'vibration' in param:
                # Vibration has more noise
                data[param] = np.random.uniform(min_val, max_val, num_samples) + \
                             np.random.normal(0, 0.02, num_samples)
            else:
                # Other parameters with some correlation to flow rate
                base_trend = np.sin(np.linspace(0, 4*np.pi, num_samples)) * 0.1 + 1
                data[param] = np.random.uniform(min_val, max_val, num_samples) * base_trend
        
        # Add operational variations (startup/shutdown cycles, load changes)
        self._add_operational_variations(data, num_samples)
        
        return pd.DataFrame(data)

    def _add_operational_variations(self, data, num_samples):
        """Add realistic operational patterns"""
        # Simulate daily load variations
        daily_pattern = np.sin(np.linspace(0, 2*np.pi*len(data['timestamp'])/96, len(data['timestamp']))) * 0.05 + 1
        
        for param in ['flow_rate', 'discharge_pressure', 'motor_current']:
            if param in data:
                data[param] = np.array(data[param]) * daily_pattern
        
        # Add random maintenance shutdowns (2-3 per year)
        maintenance_events = np.random.choice(num_samples, size=3, replace=False)
        for event_start in maintenance_events:
            event_duration = np.random.randint(4, 24)  # 1-6 hours
            event_end = min(event_start + event_duration, num_samples)
            
            for i in range(event_start, event_end):
                data['status'][i] = 'Maintenance'
                # Set all readings to near zero during maintenance
                for param in self.normal_ranges.keys():
                    if param in data:
                        data[param][i] = data[param][i] * 0.1

    def inject_failure_mode(self, baseline_data, failure_type, start_day, progression_days=30):
        """Inject gradual failure progression"""
        df = baseline_data.copy()
        start_idx = int(start_day * 24 * 60 / self.sampling_interval.total_seconds() * 60)
        end_idx = min(start_idx + int(progression_days * 24 * 60 / self.sampling_interval.total_seconds() * 60), 
                     len(df))
        
        if failure_type not in self.failure_modes:
            raise ValueError(f"Unknown failure mode: {failure_type}")
        
        failure_signature = self.failure_modes[failure_type]
        progression_factor = np.linspace(0, 1, end_idx - start_idx)
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            # Update status
            progress = progression_factor[i]
            if progress < 0.3:
                df.loc[idx, 'status'] = 'Early_Warning'
            elif progress < 0.7:
                df.loc[idx, 'status'] = 'Degrading'
            else:
                df.loc[idx, 'status'] = 'Critical'
            
            # Apply failure signature with progression
            for param, multiplier in failure_signature.items():
                if param in df.columns:
                    if multiplier > 1:  # Increasing parameter
                        df.loc[idx, param] *= (1 + (multiplier - 1) * progress)
                    else:  # Decreasing parameter
                        df.loc[idx, param] *= (multiplier + (1 - multiplier) * (1 - progress))
        
        # Add failure mode label
        df.loc[start_idx:end_idx-1, 'failure_mode'] = failure_type
        df['failure_mode'] = df['failure_mode'].fillna('None')
        
        return df

    def add_sensor_noise_and_faults(self, df, noise_level=0.02, fault_probability=0.001):
        """Add realistic sensor noise and occasional faults"""
        df_noisy = df.copy()
        
        for param in self.normal_ranges.keys():
            if param in df_noisy.columns:
                # Add Gaussian noise
                noise = np.random.normal(0, df_noisy[param].std() * noise_level, len(df_noisy))
                df_noisy[param] += noise
                
                # Occasional sensor faults (stuck values, spikes)
                fault_mask = np.random.random(len(df_noisy)) < fault_probability
                if fault_mask.any():
                    # Random fault types
                    for idx in np.where(fault_mask)[0]:
                        fault_type = np.random.choice(['spike', 'stuck', 'drift'])
                        
                        if fault_type == 'spike':
                            df_noisy.loc[idx, param] *= np.random.uniform(1.5, 3.0)
                        elif fault_type == 'stuck':
                            # Stuck sensor for 2-10 readings
                            stuck_duration = np.random.randint(2, 11)
                            stuck_value = df_noisy.loc[idx, param]
                            end_idx = min(idx + stuck_duration, len(df_noisy))
                            df_noisy.iloc[idx:end_idx, df_noisy.columns.get_loc(param)] = stuck_value
                        elif fault_type == 'drift':
                            # Gradual drift over 10-50 readings
                            drift_duration = np.random.randint(10, 51)
                            drift_magnitude = np.random.uniform(-0.2, 0.2)
                            end_idx = min(idx + drift_duration, len(df_noisy))
                            actual_duration = end_idx - idx
                            if actual_duration > 0:
                                drift_factor = np.linspace(0, drift_magnitude, actual_duration)
                                df_noisy.iloc[idx:end_idx, df_noisy.columns.get_loc(param)] *= (1 + drift_factor)
        
        return df_noisy

    def generate_complete_dataset(self, duration_days=365, num_failure_modes=5):
        """Generate complete dataset with multiple failure scenarios"""
        # Generate baseline normal operation
        print("Generating baseline normal operation data...")
        baseline_df = self.generate_baseline_data(duration_days)
        
        # Inject multiple failure modes at different times
        failure_types = list(self.failure_modes.keys())
        np.random.shuffle(failure_types)
        
        current_df = baseline_df.copy()
        
        for i in range(min(num_failure_modes, len(failure_types))):
            failure_type = failure_types[i]
            start_day = np.random.randint(30, duration_days - 60)
            progression_days = np.random.randint(20, 45)
            
            print(f"Injecting {failure_type} starting at day {start_day}")
            current_df = self.inject_failure_mode(current_df, failure_type, start_day, progression_days)
        
        # Add sensor noise and faults
        print("Adding sensor noise and faults...")
        final_df = self.add_sensor_noise_and_faults(current_df)
        
        # Calculate derived features
        final_df['differential_pressure'] = final_df['discharge_pressure'] - final_df['suction_pressure']
        final_df['vibration_magnitude'] = np.sqrt(final_df['vibration_x']**2 + 
                                                 final_df['vibration_y']**2 + 
                                                 final_df['vibration_z']**2)
        final_df['efficiency_indicator'] = (final_df['flow_rate'] * final_df['differential_pressure']) / final_df['motor_current']
        
        # Add time-based features
        final_df['hour'] = final_df['timestamp'].dt.hour
        final_df['day_of_week'] = final_df['timestamp'].dt.dayofweek
        final_df['month'] = final_df['timestamp'].dt.month
        
        return final_df

    def save_dataset(self, df, filename='pump_sensor_data.csv'):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nStatus distribution:")
        print(df['status'].value_counts())
        if 'failure_mode' in df.columns:
            print(f"\nFailure mode distribution:")
            print(df['failure_mode'].value_counts())

# Usage example
if __name__ == "__main__":
    # Initialize generator
    generator = CentrifugalPumpDataGenerator(start_date='2023-01-01', sampling_interval_minutes=15)
    
    # Generate complete dataset
    pump_data = generator.generate_complete_dataset(duration_days=365, num_failure_modes=3)
    
    # Save dataset
    generator.save_dataset(pump_data, 'centrifugal_pump_sensor_data.csv')
    
    # Display sample data
    print("\nSample data:")
    print(pump_data.head(10))
    
    # Basic statistics
    print("\nBasic statistics:")
    print(pump_data.describe())
    
    # Visualization (optional - uncomment if running in environment with matplotlib)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot key parameters over time
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    params_to_plot = ['discharge_pressure', 'vibration_magnitude', 'bearing_temp', 
                     'motor_current', 'flow_rate', 'seal_leak_rate']
    
    for i, param in enumerate(params_to_plot):
        ax = axes[i//2, i%2]
        ax.plot(pump_data['timestamp'], pump_data[param])
        ax.set_title(f'{param.replace("_", " ").title()}')
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Status distribution over time
    plt.figure(figsize=(12, 6))
    status_over_time = pump_data.groupby([pump_data['timestamp'].dt.date, 'status']).size().unstack(fill_value=0)
    status_over_time.plot(kind='area', stacked=True, alpha=0.7)
    plt.title('Equipment Status Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Readings')
    plt.legend(title='Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    """
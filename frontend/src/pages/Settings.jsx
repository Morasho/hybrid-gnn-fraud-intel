import { useState } from 'react';
import { User, Bell, Shield, Database, Mail, Key } from 'lucide-react';

export default function Settings() {
  const [activeTab, setActiveTab] = useState('Profile');
  const [thresholds, setThresholds] = useState({ high: 70, medium: 40 });
  const [toggles, setToggles] = useState({
    highRisk: true,
    daily: true,
    system: false
  });

  const menuItems = [
    { name: 'Profile', icon: User },
    { name: 'Notifications', icon: Bell },
    { name: 'Security', icon: Shield },
    { name: 'Data & Privacy', icon: Database },
    { name: 'Email Preferences', icon: Mail },
    { name: 'API Keys', icon: Key },
  ];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500">Configure your fraud detection system</p>
      </div>

      <div className="flex flex-col md:flex-row gap-8">
        
        {/* LEFT INTERNAL SIDEBAR */}
        <div className="w-full md:w-64 shrink-0">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 flex flex-col gap-2">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.name;
              return (
                <button
                  key={item.name}
                  onClick={() => setActiveTab(item.name)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg font-medium text-sm transition-colors ${
                    isActive 
                    ? 'bg-indigo-50 text-brandPrimary' 
                    : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon size={18} /> {item.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* RIGHT CONTENT AREA */}
        <div className="flex-1 space-y-6">
          
          {/* PROFILE INFORMATION CARD */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Profile Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
                <input type="text" defaultValue="Imbeka" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brandPrimary outline-none text-gray-900" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
                <input type="text" defaultValue="Musa" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brandPrimary outline-none text-gray-900" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input type="email" defaultValue="analyst@fraudguard.com" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brandPrimary outline-none text-gray-900" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <input type="text" defaultValue="Senior Fraud Analyst" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brandPrimary outline-none text-gray-900" />
              </div>
            </div>
            
            <div className="flex gap-3">
              <button className="bg-brandPrimary hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                Save Changes
              </button>
              <button className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-2 rounded-lg font-medium transition-colors">
                Cancel
              </button>
            </div>
          </div>

          {/* DETECTION THRESHOLDS CARD */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Detection Thresholds</h2>
            
            <div className="space-y-8">
              <div>
                <div className="flex justify-between items-center mb-4">
                  <label className="text-sm font-medium text-gray-700">High Risk Threshold</label>
                  <span className="font-bold text-gray-900">{thresholds.high}</span>
                </div>
                <input 
                  type="range" min="0" max="100" 
                  value={thresholds.high}
                  onChange={(e) => setThresholds({...thresholds, high: e.target.value})}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brandPrimary"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-2">
                  <span>0</span><span>100</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-4">
                  <label className="text-sm font-medium text-gray-700">Medium Risk Threshold</label>
                  <span className="font-bold text-gray-900">{thresholds.medium}</span>
                </div>
                <input 
                  type="range" min="0" max="100" 
                  value={thresholds.medium}
                  onChange={(e) => setThresholds({...thresholds, medium: e.target.value})}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brandPrimary"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-2">
                  <span>0</span><span>100</span>
                </div>
              </div>
            </div>
          </div>

          {/* NOTIFICATION PREFERENCES CARD */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-6">Notification Preferences</h2>
            
            <div className="space-y-6">
              {/* Toggle 1 */}
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium text-gray-900">High-risk transaction alerts</p>
                  <p className="text-sm text-gray-500">Get notified when high-risk transactions are detected</p>
                </div>
                <button 
                  onClick={() => setToggles({...toggles, highRisk: !toggles.highRisk})}
                  className={`w-12 h-6 rounded-full transition-colors relative ${toggles.highRisk ? 'bg-brandPrimary' : 'bg-gray-200'}`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-transform ${toggles.highRisk ? 'left-7' : 'left-1'}`}></div>
                </button>
              </div>

              {/* Toggle 2 */}
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium text-gray-900">Daily summary reports</p>
                  <p className="text-sm text-gray-500">Receive daily fraud detection summaries</p>
                </div>
                <button 
                  onClick={() => setToggles({...toggles, daily: !toggles.daily})}
                  className={`w-12 h-6 rounded-full transition-colors relative ${toggles.daily ? 'bg-brandPrimary' : 'bg-gray-200'}`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-transform ${toggles.daily ? 'left-7' : 'left-1'}`}></div>
                </button>
              </div>

              {/* Toggle 3 */}
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium text-gray-900">System status updates</p>
                  <p className="text-sm text-gray-500">Get notified about system maintenance and updates</p>
                </div>
                <button 
                  onClick={() => setToggles({...toggles, system: !toggles.system})}
                  className={`w-12 h-6 rounded-full transition-colors relative ${toggles.system ? 'bg-brandPrimary' : 'bg-gray-200'}`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-transform ${toggles.system ? 'left-7' : 'left-1'}`}></div>
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
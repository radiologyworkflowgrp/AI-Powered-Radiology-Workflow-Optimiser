import React, { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getLogs } from "@/services/api";
import { Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

const Log = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const data = await getLogs(100); // Fetch 100 logs for the full page
      setLogs(data.logs || []);
      setError(null);
    } catch (err) {
      console.error("Error fetching logs:", err);
      setError("Failed to load logs. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();

    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchLogs, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">System Logs</h1>
            <p className="text-muted-foreground">
              Real-time monitoring of system activities and ML processes
            </p>
          </div>
          <Button onClick={fetchLogs} variant="outline" size="sm">
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Activity Log</CardTitle>
            <CardDescription>
              Displaying the last 100 system events
            </CardDescription>
          </CardHeader>
          <CardContent>
            {error && (
              <div className="bg-destructive/15 text-destructive p-4 rounded-md mb-4">
                {error}
              </div>
            )}

            <div className="rounded-md border">
              <table className="w-full text-sm text-left">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="p-3 font-medium w-16">S.No</th>
                    <th className="p-3 font-medium">Patient Name</th>
                    <th className="p-3 font-medium">Status</th>
                    <th className="p-3 font-medium">Time</th>
                    <th className="p-3 font-medium">Type</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {loading && logs.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="p-8 text-center text-muted-foreground">
                        <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                        Loading logs...
                      </td>
                    </tr>
                  ) : logs.length > 0 ? (
                    logs.map((log, index) => {
                      // Extract patient name from different formats:
                      // New format: "name | status"
                      // Old format: "New patient added: name"
                      let namePart = '-';
                      let statusPart = log.action?.replace('_', ' ') || '-';

                      const desc = log.description || '';
                      if (desc.includes(' | ')) {
                        const parts = desc.split(' | ');
                        namePart = parts[0];
                        statusPart = parts[1] || statusPart;
                      } else if (desc.includes(': ')) {
                        // Old format like "New patient added: bhooi"
                        namePart = desc.split(': ').pop();
                      } else {
                        namePart = log.metadata?.patientName || log.metadata?.doctorName || log.email || desc || '-';
                      }

                      return (
                        <tr key={log.id || Math.random()} className="hover:bg-muted/50 transition-colors">
                          <td className="p-3 font-mono text-center">
                            {log.serialNumber || log.serial_number || index + 1}
                          </td>
                          <td className="p-3 font-medium">{namePart}</td>
                          <td className="p-3">
                            <Badge
                              variant="outline"
                              className={`capitalize ${statusPart.toLowerCase().includes('success') || statusPart.toLowerCase().includes('available')
                                ? 'border-green-500 text-green-500 bg-green-50 dark:bg-green-950/20'
                                : statusPart.toLowerCase().includes('error') || statusPart.toLowerCase().includes('failed')
                                  ? 'border-red-500 text-red-500 bg-red-50 dark:bg-red-950/20'
                                  : 'border-blue-500 text-blue-500 bg-blue-50 dark:bg-blue-950/20'
                                }`}
                            >
                              {statusPart}
                            </Badge>
                          </td>
                          <td className="p-3 text-muted-foreground whitespace-nowrap text-sm">
                            {new Date(log.created_at || log.createdAt || log.loginTime).toLocaleString()}
                          </td>
                          <td className="p-3">
                            <Badge variant="secondary" className="capitalize">
                              {log.entityType || log.logType || 'system'}
                            </Badge>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan="5" className="p-8 text-center text-muted-foreground">
                        No activity logs found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default Log;
import { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getLogs } from "../../services/api";

const Logs = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all');

    const fetchLogs = async () => {
        try {
            setLoading(true);
            const response = await getLogs(100, filter);
            if (response.success) {
                setLogs(response.logs);
            }
        } catch (error) {
            console.error("Error fetching logs:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLogs();
    }, [filter]);

    const getActionBadgeVariant = (action) => {
        switch (action) {
            case 'patient_added':
                return 'default';
            case 'doctor_assigned':
                return 'secondary';
            case 'report_generated':
                return 'outline';
            case 'priority_updated':
                return 'destructive';
            default:
                return 'default';
        }
    };

    return (
        <DashboardLayout role="admin">
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold">Activity Logs</h1>
                        <p className="text-muted-foreground">Track all system activities</p>
                    </div>
                    <Button onClick={fetchLogs}>Refresh</Button>
                </div>

                <div className="flex gap-2">
                    <Button
                        variant={filter === 'all' ? 'default' : 'outline'}
                        onClick={() => setFilter('all')}
                    >
                        All
                    </Button>
                    <Button
                        variant={filter === 'activity' ? 'default' : 'outline'}
                        onClick={() => setFilter('activity')}
                    >
                        Activities
                    </Button>
                    <Button
                        variant={filter === 'login' ? 'default' : 'outline'}
                        onClick={() => setFilter('login')}
                    >
                        Logins
                    </Button>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle>System Activity Log</CardTitle>
                        <CardDescription>Recent system activities and events</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <p className="text-muted-foreground">Loading logs...</p>
                        ) : logs.length === 0 ? (
                            <p className="text-muted-foreground">No logs available</p>
                        ) : (
                            <div className="space-y-3">
                                {logs.map((log) => (
                                    <div key={log._id} className="border-b pb-3 last:border-0">
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <Badge variant={log.type === 'login' ? 'default' : 'secondary'}>
                                                        {log.type}
                                                    </Badge>
                                                    {log.action && (
                                                        <Badge variant={getActionBadgeVariant(log.action)}>
                                                            {log.action.replace('_', ' ')}
                                                        </Badge>
                                                    )}
                                                    {log.role && (
                                                        <Badge variant="outline">{log.role}</Badge>
                                                    )}
                                                </div>
                                                <p className="text-sm font-medium">{log.message}</p>
                                                {log.email && (
                                                    <p className="text-xs text-muted-foreground mt-1">Email: {log.email}</p>
                                                )}
                                                {log.ipAddress && (
                                                    <p className="text-xs text-muted-foreground">IP: {log.ipAddress}</p>
                                                )}
                                                {log.metadata && log.metadata.patientName && (
                                                    <p className="text-xs text-muted-foreground">
                                                        Patient: {log.metadata.patientName}
                                                        {log.metadata.age && ` (Age: ${log.metadata.age})`}
                                                        {log.metadata.priority && ` - Priority: ${log.metadata.priority}`}
                                                    </p>
                                                )}
                                                {log.metadata && log.metadata.doctorName && (
                                                    <p className="text-xs text-muted-foreground">
                                                        Doctor: {log.metadata.doctorName}
                                                    </p>
                                                )}
                                            </div>
                                            <span className="text-xs text-muted-foreground whitespace-nowrap ml-4">
                                                {new Date(log.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </DashboardLayout>
    );
};

export default Logs;

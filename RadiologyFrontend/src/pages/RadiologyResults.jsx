import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

import { useEffect, useState } from "react";
import { getRadiologyResults } from "@/services/api"; // Import our new API function
import DicomViewer from "@/components/DicomViewer";

const RadiologyResults = () => {
  // State to store results, loading status, and errors
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedScanId, setSelectedScanId] = useState(null);

  // useEffect to fetch data when the component mounts
  useEffect(() => {
    const fetchResults = async () => {
      try {
        setLoading(true);
        const data = await getRadiologyResults();
        setResults(data);
        console.log("RADIOLOGY RESULTS FROM BACKEND:", data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchResults();
  }, []); // The empty dependency array means this effect runs once on mount

  if (loading) {
    return (
      <DashboardLayout title="Radiology Results">
        <p>Loading results...</p>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout title="Radiology Results">
        <p className="text-red-500">Error: {error}</p>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Radiology Results">
      <div className="grid gap-6">
        {results.length === 0 ? (
          <Card className="border border-slate-200 bg-slate-50">
            <CardContent className="flex flex-col items-center justify-center py-10 px-8">
              <div className="text-center space-y-3">
                <div className="w-14 h-14 bg-slate-200 rounded-full flex items-center justify-center mx-auto">
                  <svg className="w-6 h-6 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-base font-semibold text-slate-700">No Reports Available</h3>
                  <p className="text-sm text-slate-500 mt-1">Results will display here once they are ready.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ) : (
          results.map((result) => (
            <Card key={result.id || result._id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle>
                      {result.type === 'ml_generated' ? result.report_type : result.type || 'Radiology Report'}
                    </CardTitle>
                    <CardDescription>
                      {result.created_at ? new Date(result.created_at).toLocaleDateString() : result.date} -
                      {result.ml_model ? ` ML Model: ${result.ml_model}` : ` ${result.radiologist || 'Radiologist'}`}
                    </CardDescription>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Badge variant={result.status_display === 'Completed' ? 'default' : 'secondary'}>
                      {result.status_display || result.status}
                    </Badge>
                    {result.type === 'ml_generated' && result.confidence_score && (
                      <Badge variant="outline" className="text-xs">
                        Confidence: {(result.confidence_score * 100).toFixed(1)}%
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">{result.status_description}</p>
                  {result.findings && (
                    <div>
                      <h4 className="font-medium text-sm">Findings:</h4>
                      <p className="text-foreground">{result.findings}</p>
                    </div>
                  )}
                  {result.impression && (
                    <div>
                      <h4 className="font-medium text-sm">Impression:</h4>
                      <p className="text-foreground">{result.impression}</p>
                    </div>
                  )}
                  {result.recommendation && (
                    <div>
                      <h4 className="font-medium text-sm">Recommendation:</h4>
                      <p className="text-foreground">{result.recommendation}</p>
                    </div>
                  )}
                  {!result.findings && !result.impression && !result.recommendation && result.notes && (
                    <p className="text-foreground">{result.notes}</p>
                  )}
                </div>
                <div className="flex gap-2">
                  {result.scanId && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedScanId(result.scanId)}
                    >
                      🔬 View DICOM
                    </Button>
                  )}
                  <Button variant="outline" size="sm">
                    Download Report
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* DICOM Viewer Modal */}
      {selectedScanId && (
        <DicomViewer
          scanId={selectedScanId}
          onClose={() => setSelectedScanId(null)}
        />
      )}
    </DashboardLayout>
  );
};

export default RadiologyResults;

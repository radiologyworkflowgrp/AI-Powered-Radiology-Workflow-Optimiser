import { useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

const DoctorNotes = () => {
  const [patientName, setPatientName] = useState("");
  const [noteTitle, setNoteTitle] = useState("");
  const [noteContent, setNoteContent] = useState("");

  const handleSave = () => {
    toast.success("Note saved successfully");
    setPatientName("");
    setNoteTitle("");
    setNoteContent("");
  };

  const recentNotes = [
    { patient: "John Smith", title: "Follow-up examination", date: "2024-01-10" },
    { patient: "Sarah Johnson", title: "Initial consultation", date: "2024-01-12" },
    { patient: "Michael Brown", title: "Post-surgery review", date: "2024-01-14" },
  ];

  return (
    <DashboardLayout role="doctor" title="Medical Notes">
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Create New Note</CardTitle>
            <CardDescription>Add medical notes for your patients</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="patient-name">Patient Name</Label>
              <Input
                id="patient-name"
                placeholder="Enter patient name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="note-title">Note Title</Label>
              <Input
                id="note-title"
                placeholder="Enter note title"
                value={noteTitle}
                onChange={(e) => setNoteTitle(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="note-content">Note Content</Label>
              <Textarea
                id="note-content"
                placeholder="Enter note content..."
                className="min-h-[200px]"
                value={noteContent}
                onChange={(e) => setNoteContent(e.target.value)}
              />
            </div>
            <Button onClick={handleSave} className="w-full">
              Save Note
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Notes</CardTitle>
            <CardDescription>View your recent medical notes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentNotes.map((note, index) => (
                <div key={index} className="border-b pb-4 last:border-0">
                  <p className="font-medium">{note.patient}</p>
                  <p className="text-sm text-muted-foreground">{note.title}</p>
                  <p className="text-xs text-muted-foreground mt-1">{note.date}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
};

export default DoctorNotes;

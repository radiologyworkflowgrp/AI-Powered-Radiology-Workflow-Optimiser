import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const PatientNotes = () => {
  const notes = [
    {
      doctor: "Dr. Smith",
      title: "Follow-up examination",
      date: "2024-01-10",
      content: "Patient is recovering well from the procedure. Continue with prescribed medication.",
    },
    {
      doctor: "Dr. Johnson",
      title: "Initial consultation",
      date: "2024-01-05",
      content: "Patient presents with mild symptoms. Recommended imaging tests and follow-up in 2 weeks.",
    },
  ];

  return (
    <DashboardLayout role="patient" title="Medical History Notes">
      <div className="grid gap-6">
        {notes.map((note, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle>{note.title}</CardTitle>
              <CardDescription>
                {note.doctor} - {note.date}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-foreground">{note.content}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </DashboardLayout>
  );
};

export default PatientNotes;

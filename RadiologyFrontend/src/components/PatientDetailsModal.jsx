import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";

const PatientDetailsModal = ({ patient, isOpen, onClose }) => {
    if (!patient) return null;

    const getPriorityColor = (priority) => {
        if (priority <= 1) return "destructive";
        if (priority <= 2) return "default";
        return "secondary";
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle>Patient Details</DialogTitle>
                    <DialogDescription>Complete information for {patient.name}</DialogDescription>
                </DialogHeader>

                <div className="space-y-6">
                    {/* Basic Information */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Name</h3>
                            <p className="text-base">{patient.name}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Priority</h3>
                            <Badge variant={getPriorityColor(patient.priority)}>
                                Priority {patient.priority}
                            </Badge>
                        </div>
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Age</h3>
                            <p className="text-base">{patient.age || 'N/A'}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Gender</h3>
                            <p className="text-base">{patient.gender || 'N/A'}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Blood Type</h3>
                            <p className="text-base">{patient.blood_type || 'N/A'}</p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-sm text-muted-foreground">Status</h3>
                            <p className="text-base">{patient.status || 'N/A'}</p>
                        </div>
                    </div>

                    {/* Contact Information */}
                    {(patient.email || patient.guardian_name || patient.guardian_phone) && (
                        <div>
                            <h3 className="font-semibold mb-2">Contact Information</h3>
                            <div className="grid grid-cols-2 gap-4">
                                {patient.email && (
                                    <div>
                                        <h4 className="font-semibold text-sm text-muted-foreground">Email</h4>
                                        <p className="text-base">{patient.email}</p>
                                    </div>
                                )}
                                {patient.guardian_name && (
                                    <div>
                                        <h4 className="font-semibold text-sm text-muted-foreground">Guardian Name</h4>
                                        <p className="text-base">{patient.guardian_name}</p>
                                    </div>
                                )}
                                {patient.guardian_phone && (
                                    <div>
                                        <h4 className="font-semibold text-sm text-muted-foreground">Guardian Phone</h4>
                                        <p className="text-base">{patient.guardian_phone}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Medical Information */}
                    <div>
                        <h3 className="font-semibold mb-2">Medical Information</h3>
                        <div className="space-y-3">
                            {patient.symptoms && (
                                <div>
                                    <h4 className="font-semibold text-sm text-muted-foreground">Symptoms</h4>
                                    <p className="text-base">{patient.symptoms}</p>
                                </div>
                            )}
                            {patient.allergies && (
                                <div>
                                    <h4 className="font-semibold text-sm text-muted-foreground">Allergies</h4>
                                    <p className="text-base">{patient.allergies}</p>
                                </div>
                            )}
                            {patient.medications && (
                                <div>
                                    <h4 className="font-semibold text-sm text-muted-foreground">Medications</h4>
                                    <p className="text-base">{patient.medications}</p>
                                </div>
                            )}
                            {patient.weight && (
                                <div>
                                    <h4 className="font-semibold text-sm text-muted-foreground">Weight</h4>
                                    <p className="text-base">{patient.weight} kg</p>
                                </div>
                            )}
                            {patient.height && (
                                <div>
                                    <h4 className="font-semibold text-sm text-muted-foreground">Height</h4>
                                    <p className="text-base">{patient.height} cm</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Assigned Doctor */}
                    {patient.assignedDoctor && (
                        <div>
                            <h3 className="font-semibold mb-2">Assigned Doctor</h3>
                            <p className="text-base">{patient.assignedDoctor.name}</p>
                        </div>
                    )}

                    {/* Dates */}
                    <div className="grid grid-cols-2 gap-4">
                        {patient.date && (
                            <div>
                                <h4 className="font-semibold text-sm text-muted-foreground">Admission Date</h4>
                                <p className="text-base">{patient.date}</p>
                            </div>
                        )}
                        {patient.createdAt && (
                            <div>
                                <h4 className="font-semibold text-sm text-muted-foreground">Created At</h4>
                                <p className="text-base">{new Date(patient.createdAt).toLocaleString()}</p>
                            </div>
                        )}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default PatientDetailsModal;

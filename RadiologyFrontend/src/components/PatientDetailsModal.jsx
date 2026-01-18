import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState, useRef, useEffect } from "react";
import { createPrescription } from "@/services/api";
import { useUser } from "@/hooks/use-user";

const PatientDetailsModal = ({ patient, isOpen, onClose }) => {
    const { user } = useUser();
    // Custom Drag Logic
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStartPos = useRef({ x: 0, y: 0 });
    const initialOffset = useRef({ x: 0, y: 0 });

    // Prescription Logic
    const [showPrescriptionForm, setShowPrescriptionForm] = useState(false);
    const [prescriptionData, setPrescriptionData] = useState({
        medication: '',
        dosage: '',
        frequency: '',
        duration: ''
    });

    useEffect(() => {
        if (!isOpen) {
            setOffset({ x: 0, y: 0 });
            setShowPrescriptionForm(false);
        }
    }, [isOpen]);

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            const dx = e.clientX - dragStartPos.current.x;
            const dy = e.clientY - dragStartPos.current.y;
            setOffset({
                x: initialOffset.current.x + dx,
                y: initialOffset.current.y + dy
            });
        };

        const handleMouseUp = () => {
            setIsDragging(false);
        };

        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging]);

    const handleMouseDown = (e) => {
        // Don't drag if clicking form elements
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
        setIsDragging(true);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        initialOffset.current = { ...offset };
        e.preventDefault();
    };

    const handlePrescriptionSubmit = async (e) => {
        e.preventDefault();
        try {
            await createPrescription({
                ...prescriptionData,
                patientName: patient.name,
                prescribedBy: user ? user.name : 'Doctor'
            });

            // Mark patient as checked/completed
            const API_URL = "http://localhost:3002"; // Or use env var logic
            await fetch(`${API_URL}/api/patients/${patient.id || patient._id}/check`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checked: true })
            });

            setShowPrescriptionForm(false);
            setPrescriptionData({ medication: '', dosage: '', frequency: '', duration: '' });
            alert("Prescription sent successfully! Patient marked as checked.");

            // Ideally notify parent to refresh list, but for now just close
            onClose();
            // reload page to reflect changes? Or better, if onClose triggers a refresh in parent.
            // DoctorDashboard refreshes only on mount/user change. 
            // We might want to force a reload or pass a callback. 
            // But user didn't explicitly ask for list refresh, just "change status".
            window.location.reload();

        } catch (error) {
            console.error(error);
            alert("Failed to send prescription or update status");
        }
    };

    if (!patient) return null;

    const getPriorityColor = (priority) => {
        if (priority <= 1) return "destructive";
        if (priority <= 2) return "default";
        return "secondary";
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            {/*
                Draggable Modal Implementation:
                - top: 2rem (Safe start position)
                - transform: translate(calc(-50% + x), y) (Centering + Drag Offset)
                - Header acts as the handle
            */}
            <DialogContent
                className="max-w-2xl max-h-[85vh] flex flex-col p-6 transition-none"
                style={{
                    top: '2rem',
                    transform: `translate(calc(-50% + ${offset.x}px), ${offset.y}px)`,
                    // Disable transition during drag for smoothness, enable otherwise if needed
                    transition: isDragging ? 'none' : undefined
                }}
            >
                <DialogHeader
                    className="shrink-0 space-y-2 cursor-move bg-muted/30 -m-6 p-6 mb-0 select-none flex flex-row items-center justify-between"
                    onMouseDown={handleMouseDown}
                    title="Click and drag to move"
                >
                    <div className="pointer-events-none">
                        <DialogTitle>Patient Details</DialogTitle>
                        <DialogDescription>Complete information for {patient.name}</DialogDescription>
                    </div>
                </DialogHeader>

                <div className="flex-1 overflow-y-auto pr-2 -mr-2 mt-6">
                    {showPrescriptionForm ? (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <h3 className="text-lg font-semibold">New Prescription</h3>
                                <Button variant="ghost" onClick={() => setShowPrescriptionForm(false)}>Cancel</Button>
                            </div>
                            <form onSubmit={handlePrescriptionSubmit} className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Medication</label>
                                    <Input
                                        required
                                        value={prescriptionData.medication}
                                        onChange={(e) => setPrescriptionData({ ...prescriptionData, medication: e.target.value })}
                                        placeholder="e.g. Amoxicillin"
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Dosage</label>
                                        <Input
                                            required
                                            value={prescriptionData.dosage}
                                            onChange={(e) => setPrescriptionData({ ...prescriptionData, dosage: e.target.value })}
                                            placeholder="e.g. 500mg"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Frequency</label>
                                        <Input
                                            required
                                            value={prescriptionData.frequency}
                                            onChange={(e) => setPrescriptionData({ ...prescriptionData, frequency: e.target.value })}
                                            placeholder="e.g. Twice daily"
                                        />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Duration</label>
                                    <Input
                                        required
                                        value={prescriptionData.duration}
                                        onChange={(e) => setPrescriptionData({ ...prescriptionData, duration: e.target.value })}
                                        placeholder="e.g. 7 days"
                                    />
                                </div>
                                <Button type="submit" className="w-full">Issue Prescription</Button>
                            </form>
                        </div>
                    ) : (
                        <div className="space-y-6 pt-2">
                            <div className="flex justify-end">
                                <Button onClick={() => setShowPrescriptionForm(true)}>Write Prescription</Button>
                            </div>

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
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default PatientDetailsModal;
